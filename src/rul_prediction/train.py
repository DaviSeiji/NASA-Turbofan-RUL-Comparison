import argparse
import importlib.metadata
import logging
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import json
from contextlib import contextmanager
from math import prod
from pathlib import Path
from time import perf_counter


# Silencia logs informativos, mas mantém warnings e erros visíveis.
def configure_output():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    for name in ('tensorflow', 'absl', 'optuna'):
        logging.getLogger(name).setLevel(logging.WARNING)


configure_output()

import numpy as np
from src.rul_prediction.experiment import (
    DATA_ROOT,
    MODEL_ROOT,
    MODELS,
    PROTOCOL,
    assert_model_shape,
    check_output_root,
    exclusive_lock,
    experiment_path,
    identity,
    load_data,
    read_json,
    write_json,
)
from src.rul_prediction.utils import DATASETS, file_hash
from src.rul_prediction.models.model_builders import (
    build_cnn_lstm, build_transformer, build_tcn, build_itransformer,
)
from src.paths import ROOT
import tensorflow as tf
import keras

configure_output()

# Referência imutável do código que produziu os oito estudos antigos.
BASELINE_COMMIT = '1c677ab3ede8766e59dafc97f4f6b123bf96deea'
BASELINE_CODE_SHA256 = {
    'train.py': '4f5d47771465f4877d4510b2efbd6a51276e19a48d6d42a6406cd7ca1f49a8d5',
    'experiment.py': '993db5acfdef562ae54618cddd7bf807773ddede9cd03e79ee57bb21ee168307',
    'model_builders.py': 'aadd94a807e9689cb3f40efdee710f0bd4daab13364c6ea584cff683176c4854',
    'transformer_layers.py': '0768fbbfabf715174ad792d5f59c7076ebf1c2615b4a4df2089ebf9bda4bc10b',
}


# Recupera o código antigo do Git e confere seus bytes antes de usá-lo.
@contextmanager
def baseline_source(config):

    if config['model'] not in ('cnn_lstm', 'transformer') or config['code_sha256'] != BASELINE_CODE_SHA256:
        raise ValueError('O estudo não pertence à referência original das baselines.')

    files = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', BASELINE_COMMIT, '--', 'src'],
        cwd=ROOT, text=True,
    ).splitlines()

    if not files or any(not name.startswith('src/') or '..' in Path(name).parts for name in files):
        raise ValueError('Referência de código inválida ou ausente no Git.')

    with tempfile.TemporaryDirectory(prefix='rul-baseline-') as temporary:
        folder = Path(temporary)

        for name in files:
            content = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:{name}'], cwd=ROOT)

            # Esse arquivo usava CRLF no checkout original; o Git guardou LF.
            if name == 'src/rul_prediction/models/transformer_layers.py':
                content = content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')

            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        for name, digest in BASELINE_CODE_SHA256.items():
            matches = list((folder / 'src').rglob(name))

            if len(matches) != 1 or file_hash(matches[0]) != digest:
                raise ValueError(f'Código original incompatível: {name}.')

        yield folder


# Executa o treino final antigo com o código original e os mesmos dados e saídas.
def train_baseline_final(config, data_root, model_root, dataset, model, experiment, seeds, best_trial, epochs, patience):

    arguments = {
        'data_root': str(Path(data_root).resolve()),
        'model_root': str(Path(model_root).resolve()),
        'dataset': dataset, 'model': model, 'experiment': experiment,
        'seeds': seeds, 'best_trial': best_trial, 'epochs': epochs, 'patience': patience,
    }

    # O subprocesso importa somente o código congelado; as saídas continuam no projeto.
    bootstrap = (
        'import sys, json; from pathlib import Path; '
        'sys.path.insert(0, sys.argv[1]); '
        'from src import paths; paths.ROOT = Path(sys.argv[2]); '
        'from src.rul_prediction.train import train_final; '
        'train_final(**json.loads(sys.argv[3]))'
    )

    with baseline_source(config) as folder:
        print(f'[{dataset}/{model}] Código original: commit {BASELINE_COMMIT}', flush=True)
        subprocess.run(
            [sys.executable, '-c', bootstrap, str(folder), str(ROOT), json.dumps(arguments)],
            cwd=folder, check=True,
        )


# Consulta o campeão no banco sem alterar os trials ou seus resultados.
def best_trial_number(folder):

    config = read_json(Path(folder) / 'study_config.json')
    database = (Path(folder) / 'study.db').resolve()

    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        trials = connection.execute('SELECT number, state FROM trials').fetchall()

        if len(trials) != config['total_trials'] or any(state in ('RUNNING', 'WAITING') for _, state in trials):
            raise ValueError('Conclua todos os trials antes do treino final.')

        best = connection.execute(
            "SELECT t.number FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id "
            "WHERE t.state = 'COMPLETE' AND v.objective = 0 AND v.value IS NOT NULL "
            'ORDER BY v.value, t.number LIMIT 1'
        ).fetchone()

    if best is None:
        raise ValueError('Não existe trial completo para treino final.')

    return int(best[0])


# Escolhe os hiperparâmetros que o Optuna vai testar no modelo.
def suggest_params(trial, model):
    # Espaços históricos; todas as combinações de cabeças são divisíveis.

    if model == 'cnn_lstm':
        params = {
            'filters': trial.suggest_categorical('filters', [32, 64, 128]),
            'kernel_size': trial.suggest_int('kernel_size', 3, 5),
            'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128]),
            'cnn_layers': trial.suggest_int('cnn_layers', 1, 2),
            'lstm_layers': trial.suggest_int('lstm_layers', 1, 2),
        }
    elif model in ('transformer', 'itransformer'):
        params = {
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'ff_dim': trial.suggest_categorical('ff_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
        }
    elif model == 'tcn':
        params = {
            'filters': trial.suggest_categorical('filters', [32, 64, 128]),
            'kernel_size': trial.suggest_categorical('kernel_size', [2, 3, 5]),
            'num_blocks': trial.suggest_int('num_blocks', 4, 5),
        }
    else:
        raise ValueError('Modelo inválido.')

    params.update(
        dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1),
        learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
    )

    return params


# Chama o builder da arquitetura escolhida com os parâmetros recebidos.
def build_model(model, shape, params):

    builders = {
        'cnn_lstm': build_cnn_lstm, 'transformer': build_transformer,
        'tcn': build_tcn, 'itransformer': build_itransformer,
    }

    return builders[model](shape, params)


# Limpa a sessão e configura a seed e o determinismo do TensorFlow.
def prepare_runtime(seed):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# Limpa a sessão do TensorFlow depois de usar o modelo.
def clear_runtime():

    tf.keras.backend.clear_session()

# Configura a parada antecipada e, quando houver trial, a poda do Optuna.
def callbacks(patience, trial=None, epoch_history=None):

    result = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]

    if epoch_history is not None:
        # Guarda as métricas mesmo quando o Optuna interrompe o trial por poda.
        def record_epoch(epoch, logs):

            for name in ('loss', 'val_loss'):
                epoch_history.setdefault(name, []).append(float((logs or {}).get(name, float('nan'))))

        result.append(keras.callbacks.LambdaCallback(on_epoch_end=record_epoch))

    if trial is not None:
        from optuna_integration.tfkeras import TFKerasPruningCallback

        result.append(TFKerasPruningCallback(trial, 'val_loss'))

    return result


# Conta os parâmetros e registra o ambiente usado para comparar os tempos depois.
def training_metadata(model, data, params, seed, epochs, patience):

    trainable = sum(prod(int(size) for size in weight.shape) for weight in model.trainable_weights)
    total = int(model.count_params())
    devices = [
        {
            'name': device.name,
            'type': device.device_type,
            'description': tf.config.experimental.get_device_details(device).get('device_name'),
        }
        for device in tf.config.list_physical_devices()
    ]

    return {
        'trainable_parameters': int(trainable),
        'non_trainable_parameters': total - int(trainable),
        'total_parameters': total,
        'seed': seed,
        'batch_size': params['batch_size'],
        'max_epochs': epochs,
        'patience': patience,
        'training_samples': len(data['train']['X']),
        'validation_samples': len(data['validation']['X']),
        'timing_scope': 'model.fit including validation and callbacks; excluding build, data loading and save',
        'environment': {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python': platform.python_version(),
            'visible_devices': devices,
            'intra_op_threads': tf.config.threading.get_intra_op_parallelism_threads(),
            'inter_op_threads': tf.config.threading.get_inter_op_parallelism_threads(),
            'deterministic_ops': True,
        },
    }


# Acrescenta ao registro o número de épocas e as métricas da melhor validação.
def update_training_metrics(info, history):

    validation = np.asarray(history.get('val_loss', []), dtype=float)
    info['epochs_completed'] = len(validation)
    info['mean_epoch_seconds'] = info['fit_seconds'] / len(validation) if len(validation) else None
    valid_epochs = np.flatnonzero(np.isfinite(validation))

    if len(valid_epochs):
        best = int(valid_epochs[np.argmin(validation[valid_epochs])])
        training = history.get('loss', [])
        training_mse = float(training[best]) if best < len(training) else float('nan')
        info['best_epoch'] = best + 1
        info['training_mse_at_best_validation'] = training_mse if np.isfinite(training_mse) else None
        info['best_validation_mse'] = float(validation[best])


# Treina uma configuração e mede seu tempo, sem incluir construção ou salvamento.
def fit_once(
    data, model_name, params, seed, epochs, patience, trial=None,
    epoch_history=None, training_info=None,
):

    prepare_runtime(seed)
    epoch_history = epoch_history if epoch_history is not None else {}
    training_info = training_info if training_info is not None else {}

    try:
        train, val = data['train'], data['validation']
        model = build_model(model_name, train['X'].shape[1:], params)
        assert_model_shape(model, train['X'].shape[1:])
        training_info.update(training_metadata(model, data, params, seed, epochs, patience))
        training_callbacks = callbacks(patience, trial, epoch_history)
        history = None
        started = perf_counter()

        try:
            history = model.fit(
                train['X'],
                train['y'],
                validation_data=(val['X'], val['y']),
                epochs=epochs,
                batch_size=params['batch_size'],
                callbacks=training_callbacks,
                shuffle=True,
                verbose=0,
            ).history
        finally:
            training_info['fit_seconds'] = float(perf_counter() - started)
            update_training_metrics(training_info, history if history is not None else epoch_history)

        history = {key: [float(v) for v in values] for key, values in history.items()}

        if not history.get('val_loss') or not np.isfinite(history['val_loss']).all():
            raise ValueError('Histórico de validação vazio/não finito.')

        return model, history
    finally:
        clear_runtime()


# Mostra a época de parada e os MSEs de treino e validação da melhor época.
def print_training_summary(label, status, history, max_epochs, training_info=None):

    efficiency = ''

    if training_info:
        if 'fit_seconds' in training_info:
            efficiency += f' | tempo: {training_info["fit_seconds"]:.2f}s'
        if 'trainable_parameters' in training_info:
            efficiency += f' | parâmetros treináveis: {training_info["trainable_parameters"]}'

    validation = np.asarray(history.get('val_loss', []), dtype=float)
    valid_epochs = np.flatnonzero(np.isfinite(validation))

    if not len(valid_epochs):
        print(
            f'{label} {status} | épocas: {len(validation)}/{max_epochs} | MSE indisponível{efficiency}',
            flush=True,
        )
        return

    best = int(valid_epochs[np.argmin(validation[valid_epochs])])
    training = history.get('loss', [])
    training_mse = float(training[best]) if best < len(training) else float('nan')
    print(
        f'{label} {status} | parou na época {len(validation)}/{max_epochs}'
        f' | melhor época: {best + 1}'
        f' | MSE treino: {training_mse:.4f} | MSE validação: {validation[best]:.4f}{efficiency}',
        flush=True,
    )


# Consulta as versões das bibliotecas usadas no treinamento.
def software():

    return {
        name: importlib.metadata.version(name)
        for name in ('tensorflow', 'keras', 'optuna', 'optuna-integration', 'numpy')
    }


# Reúne as configurações e os hashes que identificam o estudo.
def study_config(provenance, model, experiment, seed, trials, epochs, patience):

    if trials < 1 or epochs < 1 or patience < 0 or not 0 <= seed < 2**32:
        raise ValueError('Trials/épocas positivos, patience não negativa e seed uint32 são necessários.')

    return {
        'protocol': PROTOCOL,
        'data': provenance,
        'model': model,
        'experiment': experiment,
        'sampler_seed': seed,
        'total_trials': trials,
        'epochs': epochs,
        'patience': patience,
        'objective': 'min_validation_mse',
        'pruner': 'MedianPruner(n_warmup_steps=5)',
        'training_target': 'y_capped_125',
        'software': software(),
        'code_sha256': {
            name: file_hash(path)
            for name, path in {
                'train.py': Path(__file__),
                'experiment.py': Path(__file__).with_name('experiment.py'),
                'model_builders.py': Path(__file__).parent / 'models/model_builders.py',
                'transformer_layers.py': Path(__file__).parent / 'models/transformer_layers.py',
                'utils.py': Path(__file__).with_name('utils.py'),
                'preprocessing.py': Path(__file__).with_name('preprocessing.py'),
                'paths.py': Path(__file__).parents[1] / 'paths.py',
            }.items()
        },
    }


# Confere se o estudo existente tem a mesma configuração antes de continuar.
def ensure_config(folder, config):

    path = Path(folder) / 'study_config.json'

    if path.exists():
        if read_json(path) != config:
            raise ValueError('Estudo incompatível: dados/código/configuração mudaram. Use outro experimento.')

    else:
        if (Path(folder) / 'study.db').exists():
            raise ValueError('Banco existente sem configuração: retomada recusada.')

        write_json(path, config)


# Conta quantas tentativas ainda cabem no orçamento do estudo.
def remaining_trials(trials, total):

    if any(t.state.name in ('RUNNING', 'WAITING') for t in trials):
        raise ValueError('Trial RUNNING/WAITING: confira interrupção antes de retomar.')

    if len(trials) > total:
        raise ValueError('Estudo já excede o orçamento solicitado.')

    return total - len(trials)


# Cria ou reabre o estudo Optuna salvo no banco local.
def open_study(folder, config):

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return optuna.create_study(
        study_name='rul',
        storage='sqlite:///' + (Path(folder) / 'study.db').resolve().as_posix(),
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config['sampler_seed']),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )


# Executa as tentativas restantes e salva os parâmetros do melhor trial.
def optimize(
    data_root,
    model_root,
    dataset,
    model,
    experiment,
    seed=42,
    trials=30,
    epochs=50,
    patience=5,
    fail_interrupted=False,
):

    import optuna

    check_output_root(model_root, MODEL_ROOT)
    data, provenance = load_data(Path(data_root) / dataset, ('train', 'validation'))
    folder = experiment_path(model_root, dataset, model, experiment)
    folder.mkdir(parents=True, exist_ok=True)
    config = study_config(provenance, model, experiment, seed, trials, epochs, patience)

    with exclusive_lock(folder):
        ensure_config(folder, config) 
        study = open_study(folder, config)

        if fail_interrupted:
            from optuna.trial import TrialState

            for trial in study.trials:
                if trial.state.name == 'RUNNING':
                    study.tell(trial.number, state=TrialState.FAIL)

        remaining = remaining_trials(study.trials, trials)

        # Testa uma configuração e devolve o menor erro de validação.
        def objective(trial):

            label = f'[{dataset}/{model}] Trial {trial.number + 1}/{trials} (id {trial.number})'
            print(f'{label} iniciado', flush=True)
            epoch_history = {}
            training_info = {}

            try:
                params = suggest_params(trial, model)
                fitted, history = fit_once(
                    data, model, params, (seed + trial.number) % 2**32, epochs, patience, trial,
                    epoch_history=epoch_history, training_info=training_info,
                )
            except optuna.TrialPruned:
                print_training_summary(label, 'podado', epoch_history, epochs, training_info)
                raise
            except Exception:
                print_training_summary(label, 'falhou', epoch_history, epochs, training_info)
                raise
            finally:
                if training_info:
                    trial.set_user_attr('training', training_info)

            del fitted
            print_training_summary(label, 'concluído', history, epochs, training_info)

            return min(history['val_loss'])

        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=False)

        if not any(t.state.name == 'COMPLETE' for t in study.trials):
            raise ValueError('Nenhum trial completo; não há campeão para treino final.')

        best = {
            'config_sha256': identity(config),
            'params': study.best_params,
            'trial': study.best_trial.number,
            'validation_mse': float(study.best_value),
        }

        target = folder / f'best_trial_{study.best_trial.number}.json'

        if target.exists():
            if read_json(target) != best:
                raise ValueError('Snapshot do campeão incompatível.')

        else:
            write_json(target, best)

        print(
            f'[{dataset}/{model}] Busca concluída: {len(study.trials)}/{trials} trials'
            f' | melhor trial: id {best["trial"]} | MSE validação: {best["validation_mse"]:.4f}'
            f' | arquivo: {target.name}',
            flush=True,
        )

        return target


# Salva um gráfico do erro de treino e validação ao longo das épocas.
def plot_history(history, destination):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for key in ('loss', 'val_loss'):
        ax.plot(np.arange(1, len(history[key]) + 1), history[key], label=key)

    ax.set(xlabel='Epoch', ylabel='MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


# Treina os parâmetros escolhidos para cada seed e salva os resultados.
def train_final(data_root, model_root, dataset, model, experiment, seeds, best_trial, epochs=50, patience=5):

    check_output_root(model_root, MODEL_ROOT)

    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds devem ser únicas e uint32.')

    folder = experiment_path(model_root, dataset, model, experiment)
    config = read_json(folder / 'study_config.json')

    if best_trial is None:
        best_trial = best_trial_number(folder)

    if epochs < 1 or patience < 0 or best_trial < 0:
        raise ValueError('Épocas/trial/patience inválidos.')

    if config['code_sha256'] == BASELINE_CODE_SHA256:
        return train_baseline_final(
            config, data_root, model_root, dataset, model, experiment,
            seeds, best_trial, epochs, patience,
        )

    data, provenance = load_data(Path(data_root) / dataset, ('train', 'validation'))
    expected = study_config(
        provenance,
        model,
        experiment,
        config['sampler_seed'],
        config['total_trials'],
        config['epochs'],
        config['patience'],
    )

    if expected != config:
        raise ValueError('Configuração, código ou pré-processamento mudou desde o Optuna.')

    best = read_json(folder / f'best_trial_{best_trial}.json')

    if best['config_sha256'] != identity(config) or best['trial'] != best_trial:
        raise ValueError('Hiperparâmetros pertencem a outro estudo.')

    with exclusive_lock(folder):
        for seed in seeds:
            if (folder / f'seed_{seed}').exists():
                raise FileExistsError(f'Seed {seed} já existe; nenhum modelo será sobrescrito.')

        for seed in seeds:
            target = folder / f'seed_{seed}'
            target.mkdir()  # Pasta parcial também fica protegida em caso de falha.
            final_config = {
                **config,
                'seed': seed,
                'final_epochs': epochs,
                'final_patience': patience,
                'best': best,
            }
            write_json(target / 'config.json', final_config)
            label = f'[{dataset}/{model}] Seed {seed}'
            print(f'{label} iniciada', flush=True)
            training_info = {}
            model_obj, history = fit_once(
                data, model, best['params'], seed, epochs, patience, training_info=training_info,
            )
            print_training_summary(label, 'concluída', history, epochs, training_info)
            model_obj.save(target / 'model.keras')
            write_json(target / 'history.json', history)
            plot_history(history, target / 'history.png')
            write_json(
                target / 'manifest.json',
                {
                    'protocol': PROTOCOL,
                    'data': provenance,
                    'model': model,
                    'experiment': experiment,
                    'seed': seed,
                    'best_validation_mse': min(history['val_loss']),
                    'training': training_info,
                    'sha256': {
                        name: file_hash(target / name)
                        for name in ('model.keras', 'config.json', 'history.json', 'history.png')
                    },
                },
            )
            del model_obj


# Lê os argumentos do terminal e executa inspeção, otimização ou treino final.
def main():

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    for command in ('inspect', 'optimize', 'final'):
        p = sub.add_parser(command)
        p.add_argument('--datasets', nargs='+', choices=DATASETS, default=['FD001'])
        p.add_argument('--model', choices=MODELS, default='cnn_lstm')
        p.add_argument('--experiment', default='baseline')
        p.add_argument('--data-dir', type=Path, default=DATA_ROOT)

        if command != 'inspect':
            p.add_argument('--epochs', type=int, default=50)
            p.add_argument('--patience', type=int, default=5)

        if command == 'optimize':
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--trials', type=int, default=30, help='Orçamento TOTAL, incluindo FAIL/PRUNED.')
            p.add_argument(
                '--fail-interrupted',
                action='store_true',
                help='Marca RUNNING como FAIL após interrupção confirmada.',
            )

        if command == 'final':
            p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
            p.add_argument(
                '--best-trial',
                type=int,
                help='ID do campeão; se omitido, consulta automaticamente o estudo completo.',
            )

    args = parser.parse_args()

    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Não repita datasets.')

    for dataset in args.datasets:
        if args.command == 'inspect':
            data, provenance = load_data(args.data_dir / dataset, ('train', 'validation'))
            print(dataset, provenance, {key: value['X'].shape for key, value in data.items()})
        elif args.command == 'optimize':
            optimize(
                args.data_dir,
                MODEL_ROOT,
                dataset,
                args.model,
                args.experiment,
                args.seed,
                args.trials,
                args.epochs,
                args.patience,
                args.fail_interrupted,
            )
        else:
            train_final(
                args.data_dir,
                MODEL_ROOT,
                dataset,
                args.model,
                args.experiment,
                args.seeds,
                args.best_trial,
                args.epochs,
                args.patience,
            )


if __name__ == '__main__':
    main()

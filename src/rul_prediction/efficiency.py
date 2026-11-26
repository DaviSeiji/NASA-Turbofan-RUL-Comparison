import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.rul_prediction.experiment import (
    DATA_ROOT, MODEL_ROOT, MODELS, PROTOCOL, RESULT_ROOT, identity, read_json,
)
from src.rul_prediction.utils import DATASETS, file_hash


EXPERIMENT = 'baseline'
SEEDS = (1, 2, 3, 4, 5)
OUTPUT_DIR = RESULT_ROOT / 'comparisons/quatro_modelos_todos_fd/efficiency'
TIMING_SCOPE = 'model.fit including validation and callbacks; excluding build, data loading and save'
AVERAGED_COLUMNS = (
    'TerminalRMSE', 'FitSeconds', 'EpochsExecuted', 'SecondsPerEpoch', 'BestValidationMSE',
)


# Interrompe a consolidação quando encontra dados ausentes ou inconsistentes.
def require(condition, message):

    if not condition:
        raise ValueError(message)


# Confere os hashes dos artefatos sem carregar modelos ou executar previsões.
def verify_files(folder, manifest, names):

    require(set(manifest['sha256']) == set(names), f'{folder}: manifesto incompleto.')

    for name in names:
        require(
            file_hash(folder / name) == manifest['sha256'][name],
            f'{folder / name}: hash diferente do registrado.',
        )


# Extrai uma seed e confere o histórico, o tempo e a avaliação terminal correspondente.
def collect_seed(dataset, model, seed, study, preprocessing_hash):

    parent = MODEL_ROOT / dataset / model / EXPERIMENT
    folder = parent / f'seed_{seed}'
    manifest = read_json(folder / 'manifest.json')
    verify_files(folder, manifest, ('model.keras', 'config.json', 'history.json', 'history.png'))
    config = read_json(folder / 'config.json')
    history = read_json(folder / 'history.json')
    info = manifest['training']

    for source in (manifest, config):
        require(
            (source['model'], source['seed'], source['experiment'], source['protocol'])
            == (model, seed, EXPERIMENT, PROTOCOL)
            and source['data']['Dataset'] == dataset
            and source['data']['manifest_sha256'] == preprocessing_hash,
            f'{folder}: identidade ou dados incompatíveis.',
        )

    require(manifest['data'] == config['data'], f'{folder}: proveniências dos dados diferentes.')
    require(
        all(config.get(key) == value for key, value in study.items()),
        f'{folder}: configuração diferente do estudo.',
    )
    best = config['best']
    require(
        best['config_sha256'] == identity(study)
        and best == read_json(parent / f'best_trial_{best["trial"]}.json'),
        f'{folder}: campeão diferente do estudo salvo.',
    )

    loss = np.asarray(history['loss'], dtype=float)
    validation = np.asarray(history['val_loss'], dtype=float)
    epochs = len(loss)
    require(1 <= epochs <= 50, f'{folder}: número de épocas fora de 1–50.')
    require(
        all(np.asarray(values).shape == (epochs,) and np.isfinite(values).all()
            for values in history.values()),
        f'{folder}: histórico vazio, não finito ou com comprimentos diferentes.',
    )
    require(
        config['final_epochs'] == 50 and epochs <= config['final_epochs']
        and info['max_epochs'] == config['final_epochs']
        and info['patience'] == config['final_patience']
        and info['seed'] == seed and info['batch_size'] == best['params']['batch_size'],
        f'{folder}: configuração de execução inconsistente.',
    )

    fit_seconds = info['fit_seconds']
    require(
        isinstance(fit_seconds, (int, float)) and not isinstance(fit_seconds, bool)
        and math.isfinite(fit_seconds) and fit_seconds > 0,
        f'{folder}: tempo deve ser positivo e finito.',
    )
    require(info['timing_scope'] == TIMING_SCOPE, f'{folder}: escopo do tempo diferente.')

    # Calcula pela duração do fit e pelas épocas reais, antes de agregar as seeds.
    seconds_per_epoch = fit_seconds / epochs

    if 'epochs_completed' in info:
        require(info['epochs_completed'] == epochs, f'{folder}: épocas diferem do histórico.')
    else:
        print(f'Observação: {folder}: epochs_completed ausente; usando o histórico.')

    if 'mean_epoch_seconds' in info:
        require(
            math.isclose(info['mean_epoch_seconds'], seconds_per_epoch, rel_tol=1e-12),
            f'{folder}: tempo médio por época inconsistente.',
        )
    else:
        print(f'Observação: {folder}: mean_epoch_seconds ausente; média calculada pelo histórico.')

    best_validation = float(validation.min())
    best_epoch = int(validation.argmin()) + 1
    require(
        manifest['best_validation_mse'] == best_validation
        and info['best_validation_mse'] == best_validation,
        f'{folder}: melhor MSE de validação difere do histórico.',
    )

    if 'best_epoch' in info:
        require(info['best_epoch'] == best_epoch, f'{folder}: melhor época inconsistente.')

    if 'training_mse_at_best_validation' in info:
        require(
            info['training_mse_at_best_validation'] == float(loss[best_epoch - 1]),
            f'{folder}: MSE de treino da melhor época inconsistente.',
        )

    trainable, total = info['trainable_parameters'], info['total_parameters']
    require(
        type(trainable) is int and type(total) is int and 0 < trainable <= total,
        f'{folder}: quantidade de parâmetros inválida.',
    )

    if 'non_trainable_parameters' in info:
        require(
            type(info['non_trainable_parameters']) is int
            and info['non_trainable_parameters'] >= 0
            and total == trainable + info['non_trainable_parameters'],
            f'{folder}: parâmetros treináveis e não treináveis não somam o total.',
        )

    terminal = RESULT_ROOT / dataset / model / EXPERIMENT / f'seed_{seed}' / 'terminal'
    evaluation_manifest = read_json(terminal / 'manifest.json')
    verify_files(terminal, evaluation_manifest, ('metrics.json', 'predictions.csv', 'predictions.png'))
    record = read_json(terminal / 'metrics.json')
    require(
        evaluation_manifest['protocol'] == PROTOCOL and evaluation_manifest['record'] == record
        and (record['Dataset'], record['Model'], record['Seed'], record['Experiment'],
             record['Protocol'], record['Target'])
        == (dataset, model, seed, EXPERIMENT, 'terminal', 'y_original_uncapped')
        and record['ModelSHA256'] == manifest['sha256']['model.keras']
        and record['PreprocessingManifest'] == preprocessing_hash,
        f'{terminal}: avaliação não corresponde ao modelo, à seed ou aos dados.',
    )
    training_protocol = record['TrainingProtocol']
    require(
        {'software', 'code_sha256', 'final_epochs', 'final_patience'} <= set(training_protocol)
        and all(config.get(key) == value for key, value in training_protocol.items()),
        f'{terminal}: protocolo de treinamento diferente do modelo.',
    )
    rmse = record['RMSE']
    require(math.isfinite(rmse) and rmse >= 0, f'{terminal}: RMSE inválido.')

    row = {
        'Dataset': dataset, 'Model': model, 'Seed': seed,
        'TrainableParameters': trainable, 'TotalParameters': total,
        'FitSeconds': fit_seconds, 'EpochsExecuted': epochs,
        'SecondsPerEpoch': seconds_per_epoch,
        'BestValidationMSE': best_validation, 'TerminalRMSE': rmse,
    }
    environment = {
        'Dataset': dataset, 'Model': model, 'Seed': seed,
        'software': config['software'], 'environment': info['environment'],
        'batch_size': info['batch_size'], 'training_samples': info['training_samples'],
        'validation_samples': info['validation_samples'],
    }

    return row, environment


# Junta as 80 seeds e calcula média e desvio amostral por dataset e modelo.
def consolidate():

    rows, environments = [], []

    for dataset in DATASETS:
        preprocessing_hash = file_hash(DATA_ROOT / dataset / 'manifest.json')

        for model in MODELS:
            folder = MODEL_ROOT / dataset / model / EXPERIMENT
            expected = {f'seed_{seed}' for seed in SEEDS}
            actual = {path.name for path in folder.glob('seed_*') if path.is_dir()}
            require(actual == expected, f'{folder}: seeds encontradas {sorted(actual)}; esperadas {sorted(expected)}.')
            study = read_json(folder / 'study_config.json')

            for seed in SEEDS:
                try:
                    row, environment = collect_seed(dataset, model, seed, study, preprocessing_hash)
                except (KeyError, TypeError, ValueError, OSError) as error:
                    raise ValueError(f'{dataset}/{model}/seed_{seed}: {error}') from error

                rows.append(row)
                environments.append(environment)

    individual = pd.DataFrame(rows)
    require(len(individual) == 80, 'A consolidação exige exatamente 80 combinações.')
    groups = individual.groupby(['Dataset', 'Model'], sort=False)
    summary_rows = []

    for (dataset, model), group in groups:
        require(len(group) == 5 and set(group.Seed) == set(SEEDS), f'{dataset}/{model}: seeds incompletas.')
        summary = {'Dataset': dataset, 'Model': model, 'SeedCount': len(group)}

        for column in ('TrainableParameters', 'TotalParameters'):
            require(
                group[column].nunique() == 1,
                f'{dataset}/{model}: {column} varia entre seeds: {group[["Seed", column]].to_dict("records")}.',
            )
            summary[column] = int(group[column].iloc[0])

        for column in AVERAGED_COLUMNS:
            summary[column + '_mean'] = group[column].mean()
            summary[column + '_std'] = group[column].std(ddof=1)

        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    require(len(summary) == 16, 'A consolidação exige exatamente 16 grupos.')

    return individual, summary, environments


# Mostra os ambientes registrados e diferenças que podem afetar os tempos.
def report_environments(environments):

    print('\nAmbientes registrados (dispositivo visível não comprova uso efetivo):')

    for section, keys in (
        ('software', ('tensorflow', 'keras', 'optuna', 'optuna-integration', 'numpy')),
        ('environment', ('platform', 'processor', 'python', 'visible_devices',
                         'intra_op_threads', 'inter_op_threads', 'deterministic_ops')),
    ):
        for key in keys:
            variants = {}

            for item in environments:
                if key not in item[section]:
                    print(f'Campo ausente: {section}.{key} em {item["Dataset"]}/{item["Model"]}/seed_{item["Seed"]}.')
                    continue

                value = json.dumps(item[section][key], sort_keys=True, ensure_ascii=False)
                variants[value] = variants.get(value, 0) + 1

            label = 'DIFERENÇA DE AMBIENTE' if len(variants) > 1 else 'Registrado'
            print(f'{label}: {section}.{key}: {variants}')

    print('\nBatch size por combinação (pode diferir por escolha do Optuna):')

    for dataset in DATASETS:
        for model in MODELS:
            selected = [item for item in environments if item['Dataset'] == dataset and item['Model'] == model]

            for key in ('batch_size', 'training_samples', 'validation_samples'):
                require(len({item[key] for item in selected}) == 1, f'{dataset}/{model}: {key} difere entre seeds.')

            print(f'{dataset}/{model}: batch_size={selected[0]["batch_size"]}')

    print('Processor vazio significa que o modelo da CPU não foi informado.')
    print('Threads=0 é a configuração automática, não uma medição da quantidade de threads usadas.')
    print('Não há registro de utilização efetiva da GPU, concorrência, clocks ou estado TF32 nestes campos.')


# Grava somente os dois CSVs; uma repetição idêntica preserva os arquivos existentes.
def save_tables(individual, summary):

    tables = {
        OUTPUT_DIR / 'efficiency_individual.csv': individual.to_csv(index=False, lineterminator='\n'),
        OUTPUT_DIR / 'efficiency_summary.csv': summary.to_csv(index=False, lineterminator='\n'),
    }

    for path, content in tables.items():
        if path.exists():
            require(path.read_bytes() == content.encode('utf-8'), f'{path}: saída existente diferente; não será sobrescrita.')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, content in tables.items():
        if not path.exists():
            with path.open('x', encoding='utf-8', newline='') as stream:
                stream.write(content)

        print(f'CSV: {path}')


# Executa a consolidação usando apenas artefatos existentes, sem TensorFlow ou inferência.
def main():

    individual, summary, environments = consolidate()
    report_environments(environments)
    save_tables(individual, summary)
    print('\nValidados: 80 seeds, 16 combinações e parâmetros idênticos nas cinco seeds de cada grupo.')
    print(f'Escopo do tempo: {TIMING_SCOPE}')
    print('SecondsPerEpoch = FitSeconds / EpochsExecuted, calculado antes de agregar as seeds.')
    print('Desvio-padrão amostral: ddof=1. Nenhuma média mistura datasets.')
    print('\nResumo (arredondado somente para exibição; CSVs mantêm a precisão numérica):')
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()

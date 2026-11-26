import argparse
import ast
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import pandas as pd

from src.rul_prediction.efficiency import consolidate, require, verify_files
from src.rul_prediction.experiment import MODEL_ROOT, MODELS, PROTOCOL, RESULT_ROOT, read_json
from src.rul_prediction.utils import DATASETS


BASE = RESULT_ROOT / 'comparisons/quatro_modelos_todos_fd'
SEEDS = (1, 2, 3, 4, 5)
LABELS = {'cnn_lstm': 'CNN-LSTM', 'transformer': 'Transformer', 'tcn': 'TCN', 'itransformer': 'iTransformer'}
COLORS = {'cnn_lstm': '#2F6B9A', 'transformer': '#D17A22', 'tcn': '#2D8C79', 'itransformer': '#8B5BA5'}
SEED_COLORS = ('#2F6B9A', '#D17A22', '#2D8C79', '#B84F61', '#8B5BA5')
METRICS = ('RMSE', 'MAE', 'MedianAE', 'P95AE', 'MaxAE')


# Confere se uma tabela salva coincide com os artefatos usados para calculá-la.
def check_table(actual, expected, keys, path):

    require(set(actual.columns) == set(expected.columns), f'{path}: colunas incompatíveis.')
    require(not actual.duplicated(keys).any(), f'{path}: combinações duplicadas.')

    try:
        pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True)[expected.columns],
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10,
        )
    except AssertionError as error:
        raise ValueError(f'{path}: tabela difere dos artefatos originais. {error}') from error


# Lê terminal ou windows e confere manifestos, identidades e médias das cinco seeds.
def load_comparison(protocol):

    records = []

    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                folder = RESULT_ROOT / dataset / model / 'baseline' / f'seed_{seed}' / protocol
                manifest = read_json(folder / 'manifest.json')
                verify_files(folder, manifest, ('metrics.json', 'predictions.csv', 'predictions.png'))
                record = read_json(folder / 'metrics.json')
                model_folder = MODEL_ROOT / dataset / model / 'baseline' / f'seed_{seed}'
                model_manifest = read_json(model_folder / 'manifest.json')
                config = read_json(model_folder / 'config.json')
                require(
                    manifest['protocol'] == PROTOCOL and manifest['record'] == record
                    and (record['Dataset'], record['Model'], record['Seed'], record['Experiment'],
                         record['Protocol'], record['Target'])
                    == (dataset, model, seed, 'baseline', protocol, 'y_original_uncapped')
                    and record['ModelSHA256'] == model_manifest['sha256']['model.keras']
                    and record['PreprocessingManifest'] == model_manifest['data']['manifest_sha256']
                    and all(config.get(key) == value for key, value in record['TrainingProtocol'].items()),
                    f'{folder}: identidade, modelo ou protocolo incompatível.',
                )
                require(
                    all(np.isfinite(record[key]) and record[key] >= 0 for key in METRICS),
                    f'{folder}: métricas inválidas.',
                )
                records.append(record)

    individual = pd.DataFrame(records)
    path = BASE / protocol / 'individual.csv'
    saved = pd.read_csv(path)
    saved['TrainingProtocol'] = saved['TrainingProtocol'].map(ast.literal_eval)
    check_table(saved, individual, ['Dataset', 'Model', 'Seed'], path)
    summary = individual.groupby(['Dataset', 'Model', 'Protocol'])[list(METRICS)].agg(['mean', 'std'])
    summary.columns = [f'{key}_{stat}' for key, stat in summary.columns]
    summary = summary.reset_index()
    path = BASE / protocol / 'summary.csv'
    saved_summary = pd.read_csv(path)
    check_table(saved_summary, summary, ['Dataset', 'Model'], path)

    return individual, saved_summary


# Valida as fontes e junta os números das figuras sem refazer a consolidação salva.
def load_results():

    expected_individual, expected_summary, _ = consolidate()
    efficiency = pd.read_csv(BASE / 'efficiency/efficiency_individual.csv')
    summary = pd.read_csv(BASE / 'efficiency/efficiency_summary.csv')
    check_table(efficiency, expected_individual, ['Dataset', 'Model', 'Seed'], 'efficiency_individual.csv')
    check_table(summary, expected_summary, ['Dataset', 'Model'], 'efficiency_summary.csv')
    terminal, terminal_summary = load_comparison('terminal')
    _, windows_summary = load_comparison('windows')
    combined = summary.merge(
        terminal_summary[['Dataset', 'Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std']],
        on=['Dataset', 'Model'], validate='one_to_one',
    )
    require(
        np.allclose(combined.TerminalRMSE_mean, combined.RMSE_mean, rtol=1e-10)
        and np.allclose(combined.TerminalRMSE_std, combined.RMSE_std, rtol=1e-10),
        'RMSE terminal da eficiência difere da comparação terminal.',
    )
    combined = combined.drop(columns=['RMSE_mean', 'RMSE_std']).rename(
        columns={'MAE_mean': 'TerminalMAE_mean', 'MAE_std': 'TerminalMAE_std'},
    )
    windows = windows_summary[['Dataset', 'Model', 'RMSE_mean', 'RMSE_std']].rename(
        columns={'RMSE_mean': 'WindowsRMSE_mean', 'RMSE_std': 'WindowsRMSE_std'},
    )
    combined = combined.merge(windows, on=['Dataset', 'Model'], validate='one_to_one')

    # A escolha pelo teste é apenas retrospectiva para a figura pedida, não para selecionar o modelo.
    best = terminal.sort_values(['RMSE', 'Seed']).groupby(['Dataset', 'Model'], sort=False).head(1)
    best = best[['Dataset', 'Model', 'Seed', 'RMSE']].rename(
        columns={'Seed': 'BestTerminalSeed', 'RMSE': 'BestTerminalRMSE'},
    )
    combined = combined.merge(best, on=['Dataset', 'Model'], validate='one_to_one')
    histories = {
        (dataset, model, seed): read_json(
            MODEL_ROOT / dataset / model / 'baseline' / f'seed_{seed}' / 'history.json'
        )
        for dataset in DATASETS for model in MODELS for seed in SEEDS
    }
    require(len(histories) == 80 and len(combined) == 16, 'Esperados 80 históricos e 16 grupos.')

    return combined, terminal, histories


# Salva uma nova saída ou preserva um arquivo idêntico já existente.
def save_output(path, content):

    if path.exists():
        require(path.read_bytes() == content, f'{path}: saída diferente já existe; use --output-dir com outra pasta.')
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('xb') as stream:
        stream.write(content)


# Exporta a figura em PNG a 300 dpi, sem sobrescrever resultados diferentes.
def save_figure(figure, output, relative, figures):

    buffer = BytesIO()
    figure.savefig(
        buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white',
        metadata={'Software': 'NASA-Turbofan-RUL-Comparison / plot_results.py'},
    )
    plt.close(figure)
    save_output(output / relative, buffer.getvalue())
    figures.append(relative)
    print(f'Figura: {relative}', flush=True)


# Define a escala e distingue treino de validação nas curvas de MSE.
def finish_loss_axes(ax, title, histories, handles):

    positive = all(min(history[key]) > 0 for history in histories for key in ('loss', 'val_loss'))

    if positive:
        ax.set_yscale('log')

    ax.set(title=title, xlabel='Época', ylabel='MSE (escala log)' if positive else 'MSE')
    styles = [Line2D([], [], color='#404040', label='Treino', linestyle='-'),
              Line2D([], [], color='#404040', label='Validação', linestyle='--')]
    ax.legend(handles=handles + styles, ncol=2, fontsize=9)
    ax.grid(True, which='major', alpha=.22)
    ax.set_xlim(left=1)


# Gera as 16 curvas com cinco seeds e as quatro comparações retrospectivas por dataset.
def plot_losses(summary, histories, output, figures):

    for dataset in DATASETS:
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(10, 5.8), layout='constrained')
            selected, handles = [], []

            for seed, color in zip(SEEDS, SEED_COLORS):
                history = histories[dataset, model, seed]
                selected.append(history)

                for key, style in (('loss', '-'), ('val_loss', '--')):
                    ax.plot(np.arange(1, len(history[key]) + 1), history[key], color=color, linestyle=style, linewidth=1.5)

                handles.append(Line2D([], [], color=color, label=f'Seed {seed}'))

            finish_loss_axes(ax, f'{dataset} — {LABELS[model]}\nLoss de treino e validação por época', selected, handles)
            save_figure(fig, output, f'loss_curves/{dataset}_{model}_all_seeds.png', figures)

        fig, ax = plt.subplots(figsize=(11, 6.3), layout='constrained')
        selected, handles = [], []

        for model in MODELS:
            row = summary[(summary.Dataset == dataset) & (summary.Model == model)].iloc[0]
            seed = int(row.BestTerminalSeed)
            history = histories[dataset, model, seed]
            selected.append(history)

            for key, style in (('loss', '-'), ('val_loss', '--')):
                ax.plot(np.arange(1, len(history[key]) + 1), history[key], color=COLORS[model], linestyle=style, linewidth=1.7)

            handles.append(Line2D([], [], color=COLORS[model], label=f'{LABELS[model]} (seed {seed})'))

        finish_loss_axes(
            ax, f'{dataset} — Loss de treino e validação\nSeeds com menor RMSE terminal (seleção retrospectiva pelo teste)',
            selected, handles,
        )
        save_figure(fig, output, f'loss_curves/{dataset}_best_terminal_seeds.png', figures)


# Gera uma figura simples por modelo e dataset, usando apenas a melhor seed terminal.
def plot_single_seed_losses(summary, histories, output, figures):

    for row in summary.itertuples():
        seed = int(row.BestTerminalSeed)
        history = histories[row.Dataset, row.Model, seed]
        fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')

        for key, label, color, style in (
            ('loss', 'Treino', '#2F6B9A', '-'),
            ('val_loss', 'Validação', '#D17A22', '--'),
        ):
            ax.plot(
                np.arange(1, len(history[key]) + 1), history[key],
                label=label, color=color, linestyle=style, linewidth=2,
            )

        positive = min(history['loss'] + history['val_loss']) > 0

        if positive:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(ScalarFormatter())

        ax.set(
            title=f'{row.Dataset} — {LABELS[row.Model]} — Seed {seed}',
            xlabel='Época', ylabel='MSE (escala log)' if positive else 'MSE',
            xlim=(1, max(2, len(history['loss']))),
        )
        ax.legend(loc='upper right')
        ax.grid(which='major', alpha=.22)
        save_figure(fig, output, f'loss_curves/best_seed/{row.Dataset}_{row.Model}.png', figures)

    readme = (
        '# Loss individual da melhor seed\n\n'
        'Uma figura por dataset/modelo, com apenas treino (azul contínuo) e validação '
        '(laranja tracejado). Usa o histórico real, sem prolongar as curvas.\n\n'
        'A seed é escolhida pelo menor RMSE terminal, com desempate pelo menor número '
        'da seed. Essa escolha é retrospectiva pelo teste, apenas para visualização. '
        'As duas curvas são de treino e validação, não de teste. '
        'O título identifica a seed e o eixo MSE indica quando a escala é logarítmica.\n\n'
        'Para reproduzir, na raiz do projeto:\n\n'
        '```bash\npython -m src.rul_prediction.plot_results --loss-only\n```\n\n'
        + '\n'.join(f'- [{Path(path).name}]({Path(path).name})' for path in figures) + '\n'
    )
    save_output(output / 'loss_curves/best_seed/README.md', readme.encode('utf-8'))


# Desenha barras agrupadas com o desvio amostral das cinco seeds.
def grouped_bars(ax, summary, mean, std, title, ylabel, scale=1):

    positions, width = np.arange(len(DATASETS)), .19

    for index, model in enumerate(MODELS):
        rows = summary[summary.Model == model].set_index('Dataset').loc[list(DATASETS)]
        error = rows[std].to_numpy() / scale if std else None
        ax.bar(
            positions + (index - 1.5) * width, rows[mean] / scale, width,
            yerr=error, capsize=3, color=COLORS[model], label=LABELS[model],
            error_kw={'elinewidth': 1, 'capthick': 1},
        )

    ax.set(title=title, ylabel=ylabel, ylim=(0, None))
    ax.set_xticks(positions, DATASETS)
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)
    ax.margins(y=.12)


# Gera os gráficos de desempenho, duração do treino e contagens de parâmetros.
def plot_bars(summary, output, figures):

    specifications = (
        ('TerminalRMSE', 'RMSE terminal — média ± desvio entre seeds', 'RMSE (ciclos)', 'performance/terminal_rmse.png'),
        ('TerminalMAE', 'MAE terminal — média ± desvio entre seeds', 'MAE (ciclos)', 'performance/terminal_mae.png'),
        ('FitSeconds', 'Tempo do treino final — média ± desvio entre seeds', 'Tempo de model.fit() (s)', 'efficiency/fit_seconds.png'),
        ('SecondsPerEpoch', 'Tempo médio por época — média ± desvio entre seeds', 'Segundos por época (fit / épocas reais)', 'efficiency/seconds_per_epoch.png'),
    )

    for column, title, ylabel, path in specifications:
        fig, ax = plt.subplots(figsize=(11.5, 5.8), layout='constrained')
        grouped_bars(ax, summary, column + '_mean', column + '_std', title, ylabel)
        ax.legend(ncol=4, loc='upper left', fontsize=9)
        save_figure(fig, output, path, figures)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), layout='constrained', sharey=True)

    for ax, column, title in zip(axes, ('TrainableParameters', 'TotalParameters'), ('Treináveis', 'Totais')):
        grouped_bars(ax, summary, column, None, title, 'Parâmetros (mil)', scale=1000)

    axes[0].legend(ncol=2, loc='upper left', fontsize=9)
    fig.suptitle('Quantidade de parâmetros — constante nas cinco seeds de cada grupo')
    save_figure(fig, output, 'efficiency/parameters.png', figures)


# Mostra os cinco RMSEs terminais de cada grupo e uma marca para a média.
def plot_stability(individual, output, figures):

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.4), layout='constrained', sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        for index, model in enumerate(MODELS):
            values = individual[(individual.Dataset == dataset) & (individual.Model == model)].sort_values('Seed').RMSE.to_numpy()
            ax.scatter(index + np.linspace(-.16, .16, 5), values, color=COLORS[model], s=42, edgecolor='white', linewidth=.6, zorder=3)
            ax.hlines(values.mean(), index - .25, index + .25, colors='#303030', linewidth=1.5)

        ax.set(title=dataset, xticks=np.arange(4), xticklabels=[LABELS[m] for m in MODELS])
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=.22)
        ax.set_xlim(-.6, 3.6)

    axes[0].set_ylabel('RMSE terminal (ciclos)')
    fig.suptitle('Estabilidade entre seeds — cinco pontos por grupo; traço preto = média')
    save_figure(fig, output, 'stability/terminal_rmse_seeds.png', figures)


# Relaciona as médias de RMSE e custo, separando datasets para evitar rótulos sobrepostos.
def plot_tradeoffs(summary, output, figures):

    for column, xlabel, filename in (
        ('FitSeconds_mean', 'Tempo médio do treino final (s)', 'rmse_vs_fit_seconds.png'),
        ('TrainableParameters', 'Parâmetros treináveis (mil)', 'rmse_vs_trainable_parameters.png'),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        fig.subplots_adjust(left=.07, right=.80, bottom=.09, top=.88, wspace=.95, hspace=.5)

        for ax, dataset in zip(axes.flat, DATASETS):
            group = summary[summary.Dataset == dataset].sort_values('TerminalRMSE_mean')
            scale = 1000 if column == 'TrainableParameters' else 1

            for rank, row in enumerate(group.itertuples()):
                x, y = getattr(row, column) / scale, row.TerminalRMSE_mean
                ax.scatter(x, y, color=COLORS[row.Model], s=65, edgecolor='white', linewidth=.7, zorder=3)
                ax.annotate(
                    f'{dataset} - {LABELS[row.Model]}', xy=(x, y), xytext=(1.04, .18 + .21 * rank),
                    textcoords='axes fraction', fontsize=9, color=COLORS[row.Model], va='center',
                    arrowprops={'arrowstyle': '-', 'color': COLORS[row.Model], 'linewidth': .8},
                    annotation_clip=False,
                )

            spread = max(float(group.TerminalRMSE_mean.max() - group.TerminalRMSE_mean.min()), 1)
            ax.set(
                title=dataset, xlabel=xlabel, ylabel='RMSE terminal médio (ciclos)',
                xlim=(0, float(group[column].max()) / scale * 1.15),
                ylim=(float(group.TerminalRMSE_mean.min()) - .25 * spread,
                      float(group.TerminalRMSE_mean.max()) + .25 * spread),
            )
            ax.grid(alpha=.22)

        fig.suptitle('Desempenho e custo — médias das cinco seeds\nCada painel tem sua própria escala; menor é melhor nos dois eixos', fontsize=14)
        save_figure(fig, output, 'tradeoff/' + filename, figures)


# Compara os dois protocolos e mostra rankings apenas descritivos.
def plot_optional(summary, output, figures):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained', sharey=True)

    for ax, dataset in zip(axes.flat, DATASETS):
        group = summary[summary.Dataset == dataset].set_index('Model').loc[list(MODELS)]

        for offset, prefix, alpha in ((-.18, 'Terminal', 1), (.18, 'Windows', .38)):
            ax.bar(
                np.arange(4) + offset, group[prefix + 'RMSE_mean'], .34,
                yerr=group[prefix + 'RMSE_std'], capsize=2,
                color=[COLORS[m] for m in MODELS], alpha=alpha,
            )

        ax.set(title=dataset, ylabel='RMSE (ciclos)', xticks=np.arange(4), xticklabels=[LABELS[m] for m in MODELS])
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)

    # Define a escala comum só depois de incluir todos os datasets e seus desvios.
    upper = max(
        float((summary[prefix + 'RMSE_mean'] + summary[prefix + 'RMSE_std']).max())
        for prefix in ('Terminal', 'Windows')
    )
    axes[0, 0].set_ylim(0, upper * 1.12)
    axes[0, 0].legend(handles=[Patch(color='#555555', label='Terminal'), Patch(color='#555555', alpha=.38, label='Todas as janelas')], loc='upper left')
    fig.suptitle('Terminal versus todas as janelas — média ± desvio\nProtocolos com populações de avaliação diferentes')
    save_figure(fig, output, 'performance/terminal_vs_windows_rmse.png', figures)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')

    for ax, column, title in zip(axes, ('TerminalRMSE_mean', 'FitSeconds_mean', 'TrainableParameters'), ('RMSE terminal', 'Tempo de treino', 'Parâmetros treináveis')):
        ranks = summary.assign(Rank=summary.groupby('Dataset')[column].rank(method='average'))
        matrix = ranks.pivot(index='Dataset', columns='Model', values='Rank').loc[list(DATASETS), list(MODELS)].to_numpy()
        image = ax.imshow(matrix, cmap='Blues_r', vmin=1, vmax=4)
        ax.set(title=title, xticks=np.arange(4), xticklabels=[LABELS[m] for m in MODELS], yticks=np.arange(4), yticklabels=DATASETS)
        ax.tick_params(axis='x', rotation=25)

        for row in range(4):
            for col in range(4):
                ax.text(col, row, f'{matrix[row, col]:g}', ha='center', va='center', color='white' if matrix[row, col] <= 2 else '#202020', fontsize=13)

    fig.colorbar(image, ax=axes, ticks=[1, 2, 3, 4], shrink=.8, label='Posição (1 = menor valor)')
    fig.suptitle('Rankings descritivos por dataset — sem teste de significância')
    save_figure(fig, output, 'stability/rankings.png', figures)


# Salva os números completos e uma tabela legível sem depender de bibliotecas extras.
def save_tables_and_readme(summary, output, figures):

    save_output(output / 'results_summary.csv', summary.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    lines = [
        '# Resumo dos resultados', '',
        'Média ± desvio-padrão amostral (ddof=1), cinco seeds por grupo. Arredondamento apenas nesta tabela.', '',
        '| Dataset | Modelo | RMSE terminal | MAE terminal | Treino (s) | Segundos/época | Parâmetros treináveis | Parâmetros totais |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]

    for row in summary.itertuples():
        means = [f'{getattr(row, key + "_mean"):.2f} ± {getattr(row, key + "_std"):.2f}'
                 for key in ('TerminalRMSE', 'TerminalMAE', 'FitSeconds', 'SecondsPerEpoch')]
        lines.append(f'| {row.Dataset} | {LABELS[row.Model]} | ' + ' | '.join(means)
                     + f' | {row.TrainableParameters} | {row.TotalParameters} |')

    save_output(output / 'results_summary.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    readme = [
        '# Figuras finais de resultados', '',
        'Geradas apenas a partir dos artefatos salvos, sem treinamento ou inferência. PNGs a 300 dpi.', '',
        '## Como reproduzir', '', 'Na raiz do projeto, com o ambiente Python ativado:', '',
        '```bash', 'python -m src.rul_prediction.plot_results', '```', '',
        'Repetições idênticas preservam os arquivos. Para outra renderização, use `--output-dir <nova_pasta>`; arquivos diferentes não são sobrescritos.', '',
        '## Fontes e verificações', '',
        'São conferidas as 80 seeds e os hashes dos modelos, históricos e avaliações, os dois CSVs de eficiência e os CSVs individuais e consolidados de terminal e windows. Ausências ou inconsistências interrompem a geração.', '',
        'Os históricos são MSE de treino e validação, com RUL de treino limitado a 125; não são curvas de teste. As métricas de teste usam RUL original sem limite. As figuras de loss usam escala logarítmica quando todos os valores são positivos.', '',
        'Barras de erro representam desvio-padrão amostral entre as cinco seeds (ddof=1), não intervalo de confiança. As linhas de cada seed terminam na sua última época real, sem preencher ou prolongar históricos.', '',
        'O tempo é o model.fit(), incluindo validação e callbacks e o custo inicial de compilação, excluindo construção, leitura dos dados e salvamento. Segundos/época é calculado por seed antes de obter média e desvio. Não é tempo de inferência nem tempo da busca Optuna.', '',
        'Parâmetros treináveis e totais são constantes entre as cinco seeds de cada grupo; os dois tipos são apresentados no gráfico e nas tabelas.', '',
        '## Grupos de figuras', '',
        '- `loss_curves/*_all_seeds.png`: 16 gráficos, com cor por seed, linha contínua de treino e tracejada de validação.',
        '- `loss_curves/*_best_terminal_seeds.png`: quatro comparações com a seed de menor RMSE terminal de cada modelo (empates resolvidos pelo menor número da seed). É seleção retrospectiva pelo teste, apenas para visualização; não substitui a avaliação das cinco seeds e não deve justificar seleção ou ajuste de modelos.',
        '- `performance/`: RMSE e MAE terminais com desvio, além do comparativo de RMSE terminal versus todas as janelas. Esses protocolos têm populações de exemplos diferentes.',
        '- `efficiency/`: tempo total de fit, segundos por época e parâmetros treináveis/totais. Batch sizes e épocas diferem entre configurações; não são benchmarks com carga padronizada.',
        '- `stability/terminal_rmse_seeds.png`: cinco pontos por modelo/dataset; posições da esquerda para a direita representam seeds 1 a 5, e o traço preto indica a média.',
        '- `stability/rankings.png`: rankings das médias de RMSE, tempo e parâmetros, separadamente por dataset. Menor contagem de parâmetros não implica necessariamente menor tempo. Não há teste estatístico.',
        '- `tradeoff/`: 16 pontos por figura, um por dataset/modelo, divididos em quatro painéis para manter os rótulos legíveis. Os pontos são médias e os painéis usam escalas próprias; não representam médias globais entre datasets.', '',
        '## Seeds escolhidas retrospectivamente pelo RMSE terminal', '',
        '| Dataset | Modelo | Seed | RMSE terminal |', '| --- | --- | ---: | ---: |',
    ]

    for row in summary.itertuples():
        readme.append(f'| {row.Dataset} | {LABELS[row.Model]} | {row.BestTerminalSeed} | {row.BestTerminalRMSE:.6f} |')

    readme += ['', '## Arquivos gerados', '']
    readme += [f'- [{path}]({path})' for path in figures]
    readme += ['', '- [Tabela numérica completa](results_summary.csv)', '- [Tabela de apresentação](results_summary.md)', '']
    save_output(output / 'README.md', '\n'.join(readme).encode('utf-8'))


# Valida todas as entradas antes de desenhar e grava somente novas figuras e tabelas.
def main():

    parser = argparse.ArgumentParser(description='Figuras e tabelas dos resultados já salvos, sem treinar ou prever.')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'figures')
    parser.add_argument('--loss-only', action='store_true', help='Gera somente 16 curvas individuais da melhor seed terminal.')
    args = parser.parse_args()
    summary, terminal, histories = load_results()
    print('Validados: 80 históricos, 160 avaliações e todos os CSVs de comparação e eficiência.', flush=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlepad': 12, 'figure.titlesize': 14,
    })
    output, figures = args.output_dir.resolve(), []

    if args.loss_only:
        plot_single_seed_losses(summary, histories, output, figures)
        require(len(figures) == 16, 'Esperadas 16 curvas individuais.')
        print(f'Concluído: 16 curvas individuais em {output / "loss_curves/best_seed"}', flush=True)
        return

    plot_losses(summary, histories, output, figures)
    plot_bars(summary, output, figures)
    plot_stability(terminal, output, figures)
    plot_tradeoffs(summary, output, figures)
    plot_optional(summary, output, figures)
    require(len(figures) == 30, f'Esperadas 30 figuras, geradas {len(figures)}.')
    save_tables_and_readme(summary, output, figures)
    print(f'Concluído: {len(figures)} figuras, dois resumos e README em {output}', flush=True)


if __name__ == '__main__':
    main()

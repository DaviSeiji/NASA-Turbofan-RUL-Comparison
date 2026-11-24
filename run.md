# Comandos para executar o projeto

Execute no Ubuntu/WSL, na raiz do projeto. Se estiver no PowerShell, entre primeiro com `wsl`. Os comandos abaixo usam Bash.

```bash
source ~/.venvs/rul312/bin/activate
cd /mnt/c/Users/davis/Documents/vscode/NASA-Turbofan-RUL-Comparison
```

Os arquivos `train_FD00X.txt`, `test_FD00X.txt` e `RUL_FD00X.txt` dos quatro datasets devem estar em `data/raw_data/`.

## 1. Pré-processamento

Prepara os quatro datasets. A divisão é por motor: 70% treino, 15% validação e 15% reserva, com seed 42 e RUL de treino limitado a 125.

```bash
python -m src.rul_prediction.preprocessing --datasets FD001 FD002 FD003 FD004 --seed 42 --train-fraction 0.70 --validation-fraction 0.15 --calibration-fraction 0.15 --rul-cap 125
```

Saída: `data/processed/FD001/` até `data/processed/FD004/`.

## 2. Validação dos dados processados

Confere os arquivos gerados, sem refazer o pré-processamento.

```bash
python -m src.rul_prediction.preprocessing --datasets FD001 FD002 FD003 FD004 --validate-only
```

## 3. Otimização com Optuna

Copie o bloco inteiro. Ele executa as quatro buscas CNN-LSTM e depois as quatro buscas Transformer, em sequência. Cada combinação tem **30 trials totais**, até **50 épocas por trial** e **patience 5**. São 240 trials ao todo, incluindo podados e falhos. A seleção usa o menor MSE de validação.

```bash
(
    set -e
    for modelo in cnn_lstm transformer; do
        python -m src.rul_prediction.train optimize --datasets FD001 FD002 FD003 FD004 --model "$modelo" --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5
    done
)
```

Cada combinação salva seu próprio `study.db` e `best_trial_<id>.json`. O treino final abaixo consulta o melhor ID automaticamente.

## 4. Treino final com cinco seeds

Depois de concluir a etapa 3, copie o bloco inteiro. Ele consulta o melhor trial de cada combinação e treina as seeds 1, 2, 3, 4 e 5 nos quatro datasets e nos dois modelos. São **40 treinos finais**, em sequência.

```bash
(
    set -e
    for modelo in cnn_lstm transformer; do
        for dataset in FD001 FD002 FD003 FD004; do
            pasta="models/$dataset/$modelo/baseline"
            trial=$(python -c 'import sys; from pathlib import Path; import optuna; folder = Path(sys.argv[1]).resolve(); study = optuna.load_study(study_name="rul", storage="sqlite:///" + (folder / "study.db").as_posix()); print(study.best_trial.number)' "$pasta")
            python -m src.rul_prediction.train final --datasets "$dataset" --model "$modelo" --experiment baseline --best-trial "$trial" --seeds 1 2 3 4 5 --epochs 50 --patience 5
        done
    done
)
```

Os blocos das etapas 3 e 4 param se algum comando falhar.

Saída: `models/FD00X/<modelo>/baseline/`.

Em cada `seed_N/manifest.json`, o campo `training` registra o tempo em segundos,
os parâmetros treináveis/não treináveis/totais, as épocas e as métricas da melhor
validação, além do ambiente e dos dispositivos detectados. O tempo mede o `fit`,
incluindo validação e callbacks; não inclui leitura dos dados, construção do modelo
ou salvamento. A média por época inclui o custo inicial de compilação do treino.
Durante o Optuna, os mesmos registros ficam nos atributos de cada trial no `study.db`,
inclusive quando ele é podado após começar o treino. Esses dados não mudam o objetivo
da busca, que continua sendo o menor MSE de validação.

As etapas 3 e 4 já incluem os quatro datasets e os dois modelos. Conclua os dois blocos antes das avaliações.

## 5. Avaliação principal — última janela por motor

Depois de concluir todos os treinos, avalie as cinco seeds dos dois modelos nos quatro datasets. O alvo é o RUL original, sem limite de 125.

```bash
python -m src.rul_prediction.evaluation terminal --datasets FD001 FD002 FD003 FD004 --model cnn_lstm --experiment baseline --seeds 1 2 3 4 5

python -m src.rul_prediction.evaluation terminal --datasets FD001 FD002 FD003 FD004 --model transformer --experiment baseline --seeds 1 2 3 4 5
```

## 6. Avaliação complementar — todas as janelas

```bash
python -m src.rul_prediction.evaluation windows --datasets FD001 FD002 FD003 FD004 --model cnn_lstm --experiment baseline --seeds 1 2 3 4 5

python -m src.rul_prediction.evaluation windows --datasets FD001 FD002 FD003 FD004 --model transformer --experiment baseline --seeds 1 2 3 4 5
```

Saída das avaliações: `results/FD00X/<modelo>/baseline/seed_N/terminal/` ou `windows/`, com previsões, gráfico, métricas e manifesto.

## 7. Comparação dos modelos

Gera tabelas com os resultados individuais e a média/desvio padrão entre as cinco seeds. As métricas são RMSE, MAE, MedianAE, P95AE e MaxAE, separadas por dataset e modelo.

```bash
python -m src.rul_prediction.evaluation compare --datasets FD001 FD002 FD003 FD004 --experiment baseline --seeds 1 2 3 4 5 --protocol terminal --comparison baseline_todos_fd

python -m src.rul_prediction.evaluation compare --datasets FD001 FD002 FD003 FD004 --experiment baseline --seeds 1 2 3 4 5 --protocol windows --comparison baseline_todos_fd
```

Saída: `results/comparisons/baseline_todos_fd/terminal/` e `windows/`.

## Retomada e novas execuções

- Para retomar uma busca Optuna interrompida, repita o comando original da etapa 3. Os 30 trials são o orçamento total, incluindo trials podados e falhos.
- Se houver trial `RUNNING` após uma interrupção, confirme que o processo terminou antes de usar `--fail-interrupted` no mesmo comando de otimização.
- Dados processados, seeds treinadas, avaliações e comparações existentes não são sobrescritos. Se o treino final parar no meio, execute apenas os datasets, modelos e seeds que faltam: repetir o bloco inteiro para em uma seed já existente.
- Para outra campanha, substitua `baseline` por outro nome nos comandos e no caminho `pasta` da etapa 4. Escolha outro nome em `--comparison`. Os dados processados podem ser reutilizados se o protocolo for o mesmo.
- Mantenha código, dependências e configurações estáveis durante a campanha: mudanças podem impedir a retomada pelas verificações de integridade.
- Se abrir outro terminal, entre no WSL, ative o ambiente `rul312` e entre na pasta do projeto, conforme o primeiro bloco.

Este arquivo é apenas um roteiro. Os comandos não foram executados ao criar este documento.

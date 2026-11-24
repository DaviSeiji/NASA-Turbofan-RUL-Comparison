# Comandos para executar o projeto

Execute no terminal PowerShell, na raiz do projeto, usando o Python com as dependências instaladas.

```powershell
cd 'C:\Users\davis\Documents\vscode\NASA-Turbofan-RUL-Comparison'
```

Os arquivos `train_FD00X.txt`, `test_FD00X.txt` e `RUL_FD00X.txt` dos quatro datasets devem estar em `data/raw_data/`.

## 1. Pré-processamento

Prepara os quatro datasets. A divisão é por motor: 70% treino, 15% validação e 15% reserva, com seed 42 e RUL de treino limitado a 125.

```powershell
python -m src.rul_prediction.preprocessing --datasets FD001 FD002 FD003 FD004 --seed 42 --train-fraction 0.70 --validation-fraction 0.15 --calibration-fraction 0.15 --rul-cap 125
```

Saída: `data/processed/FD001/` até `data/processed/FD004/`.

## 2. Validação dos dados processados

Confere os arquivos gerados, sem refazer o pré-processamento.

```powershell
python -m src.rul_prediction.preprocessing --datasets FD001 FD002 FD003 FD004 --validate-only
```

## 3. Otimização com Optuna

Escolha um dataset. Depois de concluir as etapas 3 e 4, repita para FD002, FD003 e FD004, alterando apenas `$dataset`.

```powershell
$dataset = 'FD001'
$experimento = 'baseline'
```

Cada comando faz uma busca independente com **30 trials totais**, até **50 épocas** e **patience 5**. A seleção usa o menor MSE de validação.

```powershell
python -m src.rul_prediction.train optimize --datasets $dataset --model cnn_lstm --experiment $experimento --seed 42 --trials 30 --epochs 50 --patience 5

python -m src.rul_prediction.train optimize --datasets $dataset --model transformer --experiment $experimento --seed 42 --trials 30 --epochs 50 --patience 5
```

Anote o **id do melhor trial** mostrado ao final de cada busca. Ele pode ser diferente para cada modelo e dataset. Use o `id`, que começa em zero, e não a posição mostrada em `Trial 1/30`.

## 4. Treino final com cinco seeds

Informe os IDs dos melhores trials das buscas que acabou de concluir para o dataset escolhido:

```powershell
$cnnTrial = [int](Read-Host "ID do melhor trial CNN-LSTM para $dataset")
$transformerTrial = [int](Read-Host "ID do melhor trial Transformer para $dataset")
```

Treine as seeds 1, 2, 3, 4 e 5:

```powershell
python -m src.rul_prediction.train final --datasets $dataset --model cnn_lstm --experiment $experimento --best-trial $cnnTrial --seeds 1 2 3 4 5 --epochs 50 --patience 5

python -m src.rul_prediction.train final --datasets $dataset --model transformer --experiment $experimento --best-trial $transformerTrial --seeds 1 2 3 4 5 --epochs 50 --patience 5
```

Saída: `models/FD00X/<modelo>/baseline/`.

**Repita as etapas 3 e 4 para FD002, FD003 e FD004 antes de começar as avaliações.** Cada dataset precisa das próprias buscas e dos próprios IDs de campeões.

## 5. Avaliação principal — última janela por motor

Depois de concluir todos os treinos, avalie as cinco seeds dos dois modelos nos quatro datasets. O alvo é o RUL original, sem limite de 125.

```powershell
python -m src.rul_prediction.evaluation terminal --datasets FD001 FD002 FD003 FD004 --model cnn_lstm --experiment $experimento --seeds 1 2 3 4 5

python -m src.rul_prediction.evaluation terminal --datasets FD001 FD002 FD003 FD004 --model transformer --experiment $experimento --seeds 1 2 3 4 5
```

## 6. Avaliação complementar — todas as janelas

```powershell
python -m src.rul_prediction.evaluation windows --datasets FD001 FD002 FD003 FD004 --model cnn_lstm --experiment $experimento --seeds 1 2 3 4 5

python -m src.rul_prediction.evaluation windows --datasets FD001 FD002 FD003 FD004 --model transformer --experiment $experimento --seeds 1 2 3 4 5
```

Saída das avaliações: `results/FD00X/<modelo>/baseline/seed_N/terminal/` ou `windows/`, com previsões, gráfico, métricas e manifesto.

## 7. Comparação dos modelos

Gera tabelas com os resultados individuais e a média/desvio padrão entre as cinco seeds. As métricas são RMSE, MAE, MedianAE, P95AE e MaxAE, separadas por dataset e modelo.

```powershell
python -m src.rul_prediction.evaluation compare --datasets FD001 FD002 FD003 FD004 --experiment $experimento --seeds 1 2 3 4 5 --protocol terminal --comparison baseline_todos_fd

python -m src.rul_prediction.evaluation compare --datasets FD001 FD002 FD003 FD004 --experiment $experimento --seeds 1 2 3 4 5 --protocol windows --comparison baseline_todos_fd
```

Saída: `results/comparisons/baseline_todos_fd/terminal/` e `windows/`.

## Retomada e novas execuções

- Para retomar uma busca Optuna interrompida, repita o comando original da etapa 3. Os 30 trials são o orçamento total, incluindo trials podados e falhos.
- Se houver trial `RUNNING` após uma interrupção, confirme que o processo terminou antes de usar `--fail-interrupted` no mesmo comando de otimização.
- Dados processados, seeds treinadas, avaliações e comparações existentes não são sobrescritos. Não repita essas etapas para saídas já concluídas.
- Para outra campanha, escolha um novo `$experimento` e outro nome em `--comparison`. Os dados processados podem ser reutilizados se o protocolo for o mesmo.
- Mantenha código, dependências e configurações estáveis durante a campanha: mudanças podem impedir a retomada pelas verificações de integridade.
- Se abrir outro terminal, defina novamente `$experimento = 'baseline'` e, nas etapas 3 e 4, o `$dataset` desejado.

Este arquivo é apenas um roteiro. Os comandos não foram executados ao criar este documento.

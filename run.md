# Comandos para executar o projeto

Execute no Ubuntu/WSL, na raiz do projeto. Se estiver no PowerShell, entre primeiro com `wsl`. Os comandos abaixo usam Bash.

```bash
source ~/.venvs/rul312/bin/activate
cd /mnt/c/Users/davis/Documents/vscode/NASA-Turbofan-RUL-Comparison
```

Os arquivos `train_FD00X.txt`, `test_FD00X.txt` e `RUL_FD00X.txt` dos quatro datasets devem estar em `data/raw_data/`.

## Estudos anteriores e código utilizado

Os oito estudos de CNN-LSTM e Transformer foram conferidos antes desta alteração: todos têm 30 trials, sem RUNNING, WAITING ou FAIL. Os IDs são os números dos trials no Optuna, começando em zero.

| Dataset | Modelo | Melhor trial | MSE de validação | Completos / podados |
| --- | --- | ---: | ---: | ---: |
| FD001 | CNN-LSTM | 7 | 146.73843383789062 | 8 / 22 |
| FD001 | Transformer | 15 | 144.87969970703125 | 12 / 18 |
| FD002 | CNN-LSTM | 7 | 228.72743225097656 | 9 / 21 |
| FD002 | Transformer | 12 | 227.0980682373047 | 19 / 11 |
| FD003 | CNN-LSTM | 28 | 144.26219177246094 | 17 / 13 |
| FD003 | Transformer | 6 | 158.90048217773438 | 16 / 14 |
| FD004 | CNN-LSTM | 7 | 247.03680419921875 | 10 / 20 |
| FD004 | Transformer | 26 | 233.63299560546875 | 16 / 14 |

O código que gerou esses estudos está no commit `1c677ab3ede8766e59dafc97f4f6b123bf96deea`. O comando `final` reconhece os hashes originais e recupera esse código diretamente do Git em uma pasta temporária. Apenas `transformer_layers.py` precisa ter seus finais de linha CRLF restaurados, como eram no checkout original. Os quatro hashes são conferidos byte a byte antes da execução. Nenhum hash salvo é alterado.

O processo antigo usa os dados e as pastas de saída deste projeto e executa suas verificações originais de código, software e configuração. A pasta temporária é removida ao terminar. Não há checkout, troca de branch ou cópia permanente de outro pipeline. Preserve o histórico Git e as versões do `requirements.txt`; sem a referência original ou com software diferente, o treino antigo deve falhar.

TCN e iTransformer usam o código atual e registram seus próprios hashes no `study_config.json`. Os hashes agora também incluem `utils.py`, `preprocessing.py` e `paths.py`. Não edite esses arquivos durante a campanha. A avaliação preserva a proveniência de cada arquitetura, mas compara o protocolo experimental comum, os dados, as janelas e os alvos.

TCN: duas Conv1D causais por bloco, dilatações 1, 2, 4, 8... e conexões residuais; filtros 32/64/128, kernel 2/3/5 e 4–5 blocos. O campo receptivo é `1 + 2*(kernel-1)*(2**blocos-1)`, sempre pelo menos 31 ciclos. A cabeça usa o último ciclo, Dense(32, ReLU) e Dense(1).

iTransformer: cada variável selecionada vira um token com 30 valores, projetado para `d_model`. A atenção opera entre variáveis, sem posição temporal adicionada. Usa os mesmos valores de dimensão, cabeças e FFN do espaço Transformer, com 1–3 blocos. A cabeça com pooling entre variáveis e regressão de um RUL é uma adaptação para esta tarefa, em vez da previsão multivariada do artigo. Mantém a normalização do pré-processamento existente, sem acrescentar normalização por janela.

Referências das arquiteturas: [TCN](https://arxiv.org/abs/1803.01271), [iTransformer](https://arxiv.org/abs/2310.06625) e [implementação original do iTransformer](https://github.com/thuml/iTransformer).

## 1. Pré-processamento

Prepara os quatro datasets. A divisão é por motor: 70% treino, 15% validação e 15% reserva, com seed 42 e RUL de treino limitado a 125.

Os dados desta campanha já foram preparados. Não repita esta etapa para as pastas existentes.

```bash
python -m src.rul_prediction.preprocessing --datasets FD001 FD002 FD003 FD004 --seed 42 --train-fraction 0.70 --validation-fraction 0.15 --calibration-fraction 0.15 --rul-cap 125
```

Saída: `data/processed/FD001/` até `data/processed/FD004/`.

## 2. Validação dos dados processados

Confere os arquivos gerados, sem refazer o pré-processamento.

```bash
python -m src.rul_prediction.preprocessing --datasets FD001 FD002 FD003 FD004 --validate-only
```

## 3. Otimização com Optuna — somente os dois modelos novos

Os Optunas antigos já terminaram. Não os execute novamente com o código atual. Copie o bloco inteiro para executar as quatro buscas TCN e depois as quatro buscas iTransformer, em sequência. Cada combinação tem **30 trials totais**, até **50 épocas por trial**, **patience 5** e sampler seed 42. São 240 novos trials ao todo, incluindo podados e falhos. A seleção usa o menor MSE de validação.

```bash
(
    set -e
    for modelo in tcn itransformer; do
        python -W ignore -m src.rul_prediction.train optimize --datasets FD001 FD002 FD003 FD004 --model "$modelo" --experiment baseline --seed 42 --trials 30 --epochs 50 --patience 5 2>> results/training_stderr.log
    done
)
```

Cada combinação salva seu próprio `study.db` e `best_trial_<id>.json`. O treino final abaixo consulta o melhor ID automaticamente.

Os comandos de treino ocultam os warnings Python e enviam a saída de diagnóstico das bibliotecas para `results/training_stderr.log`, acrescentando ao arquivo a cada execução. Os resumos dos trials e seeds continuam no terminal. Se um comando falhar, o traceback também estará nesse arquivo; consulte com `tail -n 80 results/training_stderr.log`. Isso não altera o código nem a configuração dos estudos existentes. Um processo já em execução mantém a saída antiga até terminar.

## 4. Treino final com cinco seeds

Depois de concluir a etapa 3, copie o bloco inteiro. Ao omitir `--best-trial`, o próprio comando consulta o campeão de cada estudo completo. Treina as seeds 1, 2, 3, 4 e 5 nos quatro datasets e nos quatro modelos. São **80 treinos finais**, em sequência. O código antigo é selecionado automaticamente para CNN-LSTM e Transformer; não é necessário trocar versões manualmente.

```bash
(
    set -e
    for modelo in cnn_lstm transformer tcn itransformer; do
        python -W ignore -m src.rul_prediction.train final --datasets FD001 FD002 FD003 FD004 --model "$modelo" --experiment baseline --seeds 1 2 3 4 5 --epochs 50 --patience 5 2>> results/training_stderr.log
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

A etapa 3 inclui os dois modelos novos; a etapa 4 inclui os quatro modelos. Conclua os dois blocos antes das avaliações.

## 5. Avaliação principal — última janela por motor

Depois de concluir todos os treinos, avalie as cinco seeds dos quatro modelos nos quatro datasets. O alvo é o RUL original, sem limite de 125. Usa o avaliador atual para todos os modelos, sem alterar o treinamento congelado das baselines.

```bash
(
    set -e
    for modelo in cnn_lstm transformer tcn itransformer; do
        python -m src.rul_prediction.evaluation terminal --datasets FD001 FD002 FD003 FD004 --model "$modelo" --experiment baseline --seeds 1 2 3 4 5
    done
)
```

## 6. Avaliação complementar — todas as janelas

```bash
(
    set -e
    for modelo in cnn_lstm transformer tcn itransformer; do
        python -m src.rul_prediction.evaluation windows --datasets FD001 FD002 FD003 FD004 --model "$modelo" --experiment baseline --seeds 1 2 3 4 5
    done
)
```

Saída das avaliações: `results/FD00X/<modelo>/baseline/seed_N/terminal/` ou `windows/`, com previsões, gráfico, métricas e manifesto.

## 7. Comparação dos modelos

Gera tabelas com os resultados individuais e a média/desvio padrão entre as cinco seeds. As métricas são RMSE, MAE, MedianAE, P95AE e MaxAE, separadas por dataset e modelo.

```bash
(
    set -e
    for protocolo in terminal windows; do
        python -m src.rul_prediction.evaluation compare --datasets FD001 FD002 FD003 FD004 --experiment baseline --seeds 1 2 3 4 5 --protocol "$protocolo" --comparison quatro_modelos_todos_fd
    done
)
```

Saída: `results/comparisons/quatro_modelos_todos_fd/terminal/` e `windows/`.

## Retomada e novas execuções

- Para retomar uma busca Optuna interrompida, repita o comando original da etapa 3. Os 30 trials são o orçamento total, incluindo trials podados e falhos.
- Se houver trial `RUNNING` após uma interrupção, confirme que o processo terminou antes de usar `--fail-interrupted` no mesmo comando de otimização.
- Dados processados, seeds treinadas, avaliações e comparações existentes não são sobrescritos. Se o treino final parar no meio, execute apenas os datasets, modelos e seeds que faltam: repetir o bloco inteiro para em uma seed já existente.
- Para outra campanha, substitua `baseline` por outro nome nos comandos. Escolha outro nome em `--comparison`. Os dados processados podem ser reutilizados se o protocolo for o mesmo. Os estudos antigos `baseline` continuam reservados ao código original.
- Mantenha código, dependências e configurações estáveis durante a campanha: mudanças podem impedir a retomada pelas verificações de integridade.
- Se abrir outro terminal, entre no WSL, ative o ambiente `rul312` e entre na pasta do projeto, conforme o primeiro bloco.

Este arquivo é apenas um roteiro. Os comandos não foram executados ao criar este documento.

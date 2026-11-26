# Figuras finais de resultados

Geradas apenas a partir dos artefatos salvos, sem treinamento ou inferência. PNGs a 300 dpi.

## Como reproduzir

Na raiz do projeto, com o ambiente Python ativado:

```bash
python -m src.rul_prediction.plot_results
```

Repetições idênticas preservam os arquivos. Para outra renderização, use `--output-dir <nova_pasta>`; arquivos diferentes não são sobrescritos.

## Fontes e verificações

São conferidas as 80 seeds e os hashes dos modelos, históricos e avaliações, os dois CSVs de eficiência e os CSVs individuais e consolidados de terminal e windows. Ausências ou inconsistências interrompem a geração.

Os históricos são MSE de treino e validação, com RUL de treino limitado a 125; não são curvas de teste. As métricas de teste usam RUL original sem limite. As figuras de loss usam escala logarítmica quando todos os valores são positivos.

Barras de erro representam desvio-padrão amostral entre as cinco seeds (ddof=1), não intervalo de confiança. As linhas de cada seed terminam na sua última época real, sem preencher ou prolongar históricos.

O tempo é o model.fit(), incluindo validação e callbacks e o custo inicial de compilação, excluindo construção, leitura dos dados e salvamento. Segundos/época é calculado por seed antes de obter média e desvio. Não é tempo de inferência nem tempo da busca Optuna.

Parâmetros treináveis e totais são constantes entre as cinco seeds de cada grupo; os dois tipos são apresentados no gráfico e nas tabelas.

## Grupos de figuras

- `loss_curves/*_all_seeds.png`: 16 gráficos, com cor por seed, linha contínua de treino e tracejada de validação.
- `loss_curves/*_best_terminal_seeds.png`: quatro comparações com a seed de menor RMSE terminal de cada modelo (empates resolvidos pelo menor número da seed). É seleção retrospectiva pelo teste, apenas para visualização; não substitui a avaliação das cinco seeds e não deve justificar seleção ou ajuste de modelos.
- `performance/`: RMSE e MAE terminais com desvio, além do comparativo de RMSE terminal versus todas as janelas. Esses protocolos têm populações de exemplos diferentes.
- `efficiency/`: tempo total de fit, segundos por época e parâmetros treináveis/totais. Batch sizes e épocas diferem entre configurações; não são benchmarks com carga padronizada.
- `stability/terminal_rmse_seeds.png`: cinco pontos por modelo/dataset; posições da esquerda para a direita representam seeds 1 a 5, e o traço preto indica a média.
- `stability/rankings.png`: rankings das médias de RMSE, tempo e parâmetros, separadamente por dataset. Menor contagem de parâmetros não implica necessariamente menor tempo. Não há teste estatístico.
- `tradeoff/`: 16 pontos por figura, um por dataset/modelo, divididos em quatro painéis para manter os rótulos legíveis. Os pontos são médias e os painéis usam escalas próprias; não representam médias globais entre datasets.

## Seeds escolhidas retrospectivamente pelo RMSE terminal

| Dataset | Modelo | Seed | RMSE terminal |
| --- | --- | ---: | ---: |
| FD001 | CNN-LSTM | 1 | 14.490746 |
| FD001 | Transformer | 5 | 15.415204 |
| FD001 | TCN | 1 | 14.694902 |
| FD001 | iTransformer | 2 | 13.208143 |
| FD002 | CNN-LSTM | 3 | 26.138816 |
| FD002 | Transformer | 3 | 25.760413 |
| FD002 | TCN | 1 | 26.490736 |
| FD002 | iTransformer | 3 | 25.971682 |
| FD003 | CNN-LSTM | 3 | 14.961129 |
| FD003 | Transformer | 2 | 14.366250 |
| FD003 | TCN | 1 | 14.822835 |
| FD003 | iTransformer | 4 | 17.734742 |
| FD004 | CNN-LSTM | 5 | 27.423278 |
| FD004 | Transformer | 4 | 26.727281 |
| FD004 | TCN | 2 | 27.366137 |
| FD004 | iTransformer | 3 | 26.925482 |

## Arquivos gerados

- [loss_curves/FD001_cnn_lstm_all_seeds.png](loss_curves/FD001_cnn_lstm_all_seeds.png)
- [loss_curves/FD001_transformer_all_seeds.png](loss_curves/FD001_transformer_all_seeds.png)
- [loss_curves/FD001_tcn_all_seeds.png](loss_curves/FD001_tcn_all_seeds.png)
- [loss_curves/FD001_itransformer_all_seeds.png](loss_curves/FD001_itransformer_all_seeds.png)
- [loss_curves/FD001_best_terminal_seeds.png](loss_curves/FD001_best_terminal_seeds.png)
- [loss_curves/FD002_cnn_lstm_all_seeds.png](loss_curves/FD002_cnn_lstm_all_seeds.png)
- [loss_curves/FD002_transformer_all_seeds.png](loss_curves/FD002_transformer_all_seeds.png)
- [loss_curves/FD002_tcn_all_seeds.png](loss_curves/FD002_tcn_all_seeds.png)
- [loss_curves/FD002_itransformer_all_seeds.png](loss_curves/FD002_itransformer_all_seeds.png)
- [loss_curves/FD002_best_terminal_seeds.png](loss_curves/FD002_best_terminal_seeds.png)
- [loss_curves/FD003_cnn_lstm_all_seeds.png](loss_curves/FD003_cnn_lstm_all_seeds.png)
- [loss_curves/FD003_transformer_all_seeds.png](loss_curves/FD003_transformer_all_seeds.png)
- [loss_curves/FD003_tcn_all_seeds.png](loss_curves/FD003_tcn_all_seeds.png)
- [loss_curves/FD003_itransformer_all_seeds.png](loss_curves/FD003_itransformer_all_seeds.png)
- [loss_curves/FD003_best_terminal_seeds.png](loss_curves/FD003_best_terminal_seeds.png)
- [loss_curves/FD004_cnn_lstm_all_seeds.png](loss_curves/FD004_cnn_lstm_all_seeds.png)
- [loss_curves/FD004_transformer_all_seeds.png](loss_curves/FD004_transformer_all_seeds.png)
- [loss_curves/FD004_tcn_all_seeds.png](loss_curves/FD004_tcn_all_seeds.png)
- [loss_curves/FD004_itransformer_all_seeds.png](loss_curves/FD004_itransformer_all_seeds.png)
- [loss_curves/FD004_best_terminal_seeds.png](loss_curves/FD004_best_terminal_seeds.png)
- [performance/terminal_rmse.png](performance/terminal_rmse.png)
- [performance/terminal_mae.png](performance/terminal_mae.png)
- [efficiency/fit_seconds.png](efficiency/fit_seconds.png)
- [efficiency/seconds_per_epoch.png](efficiency/seconds_per_epoch.png)
- [efficiency/parameters.png](efficiency/parameters.png)
- [stability/terminal_rmse_seeds.png](stability/terminal_rmse_seeds.png)
- [tradeoff/rmse_vs_fit_seconds.png](tradeoff/rmse_vs_fit_seconds.png)
- [tradeoff/rmse_vs_trainable_parameters.png](tradeoff/rmse_vs_trainable_parameters.png)
- [performance/terminal_vs_windows_rmse.png](performance/terminal_vs_windows_rmse.png)
- [stability/rankings.png](stability/rankings.png)

- [Tabela numérica completa](results_summary.csv)
- [Tabela de apresentação](results_summary.md)

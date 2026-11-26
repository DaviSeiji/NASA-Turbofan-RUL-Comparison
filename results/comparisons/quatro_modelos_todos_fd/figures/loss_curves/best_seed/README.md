# Loss individual da melhor seed

Uma figura por dataset/modelo, com apenas treino (azul contínuo) e validação (laranja tracejado). Usa o histórico real, sem prolongar as curvas.

A seed é escolhida pelo menor RMSE terminal, com desempate pelo menor número da seed. Essa escolha é retrospectiva pelo teste, apenas para visualização. As duas curvas são de treino e validação, não de teste. O título identifica a seed e o eixo MSE indica quando a escala é logarítmica.

Para reproduzir, na raiz do projeto:

```bash
python -m src.rul_prediction.plot_results --loss-only
```

- [FD001_cnn_lstm.png](FD001_cnn_lstm.png)
- [FD001_transformer.png](FD001_transformer.png)
- [FD001_tcn.png](FD001_tcn.png)
- [FD001_itransformer.png](FD001_itransformer.png)
- [FD002_cnn_lstm.png](FD002_cnn_lstm.png)
- [FD002_transformer.png](FD002_transformer.png)
- [FD002_tcn.png](FD002_tcn.png)
- [FD002_itransformer.png](FD002_itransformer.png)
- [FD003_cnn_lstm.png](FD003_cnn_lstm.png)
- [FD003_transformer.png](FD003_transformer.png)
- [FD003_tcn.png](FD003_tcn.png)
- [FD003_itransformer.png](FD003_itransformer.png)
- [FD004_cnn_lstm.png](FD004_cnn_lstm.png)
- [FD004_transformer.png](FD004_transformer.png)
- [FD004_tcn.png](FD004_tcn.png)
- [FD004_itransformer.png](FD004_itransformer.png)

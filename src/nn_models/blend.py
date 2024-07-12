import sys
import json
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import preprocessing
import metrics


def read_predictions(model_directory, idx):

    oof_predictions = np.load(settings.MODELS / model_directory / 'oof_predictions.npy')[idx]
    test_predictions = pd.read_parquet(settings.MODELS / model_directory / 'submission.parquet')
    settings.logger.info(f'Loaded predictions from {model_directory}')

    return oof_predictions, test_predictions


if __name__ == '__main__':

    df = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    columns = df.columns.tolist()
    target_columns = columns[-368:]
    target_column_groups = preprocessing.get_target_column_groups(columns=target_columns)
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    target_weights = np.load(settings.DATA / 'datasets' / 'target_weights.npy')

    test_features = pd.read_parquet(settings.DATA / 'datasets' / 'test.parquet').to_numpy()
    np.random.seed(42)
    idx = np.random.randint(0, 10091520, 1_000_000)
    targets = df[target_columns].to_numpy()[idx, :]
    del df

    hard_targets = [
        'ptend_q0001_12', 'ptend_q0001_13',
        'ptend_q0001_14', 'ptend_q0001_15', 'ptend_q0001_16', 'ptend_q0001_17',
        'ptend_q0001_18', 'ptend_q0001_19', 'ptend_q0001_20',
        'ptend_q0002_27', 'ptend_q0002_28', 'ptend_q0002_29', 'ptend_q0002_30',
        'ptend_q0003_12', 'ptend_q0003_13', 'ptend_q0003_14', 'ptend_q0003_15',
        'ptend_q0003_16', 'ptend_q0003_17', 'ptend_q0003_18', 'ptend_q0003_19', 'ptend_q0003_20'
    ]
    hard_target_idx = np.array([target_columns.index(target) for target in hard_targets])

    unet_seresnet_1m_oof_predictions, unet_seresnet_1m_test_predictions = read_predictions('unet_seresnet_1m_5f', idx=idx)
    unet_seresnet_4m_oof_predictions, unet_seresnet_4m_test_predictions = read_predictions('unet_seresnet_4m_5f', idx=idx)
    unet_seresnet_20m_oof_predictions, unet_seresnet_20m_test_predictions = read_predictions('unet_seresnet_20m_5f', idx=idx)

    seq2seq_seresnet_1m_oof_predictions, seq2seq_seresnet_1m_test_predictions = read_predictions('seq2seq_seresnet_1m_5f', idx=idx)
    seq2seq_seresnet_4m_oof_predictions, seq2seq_seresnet_4m_test_predictions = read_predictions('seq2seq_seresnet_4m_5f', idx=idx)
    seq2seq_seresnet_20m_oof_predictions, seq2seq_seresnet_20m_test_predictions = read_predictions('seq2seq_seresnet_20m_5f', idx=idx)

    gru_10m_oof_predictions, gru_10m_test_predictions = read_predictions('gru_10m_5f', idx=idx)
    #gru_17m_oof_predictions, gru_17m_test_predictions = read_predictions('gru_17m_features')
    lstm_10m_oof_predictions, lstm_10m_test_predictions = read_predictions('lstm_10m_5f', idx=idx)
    #lstm_17m_oof_predictions, lstm_17m_test_predictions = read_predictions('lstm_17m_features')

    oof_predictions = {
        #'gru_10m': gru_10m_oof_predictions,
        #'gru_17m': gru_17m_oof_predictions,
        #'gru_18m': gru_18m_oof_predictions,
        #'lstm_10m': lstm_10m_oof_predictions,
        #'lstm_17m': lstm_17m_oof_predictions,

        #'unet_seresnet_1m': unet_seresnet_1m_oof_predictions,
        #'unet_seresnet_4m': unet_seresnet_4m_oof_predictions,
        #'unet_seresnet_20m': unet_seresnet_20m_oof_predictions,
    }
    for model, predictions in oof_predictions.items():
        global_oof_scores, _ = metrics.regression_scores(
            y_true=targets,
            y_pred=predictions,
            weights=target_weights,
            target_columns=target_columns
        )
        settings.logger.info(f'{model} OOF Scores: {json.dumps(global_oof_scores, indent=2)}')

    gru_10m_weight = 0.5
    #gru_17m_weight = 0.05
    lstm_10m_weight = 0.5
    #lstm_17m_weight = 0.5
    rnn_oof_predictions = gru_10m_oof_predictions * gru_10m_weight + \
                          lstm_10m_oof_predictions * lstm_10m_weight
    rnn_global_oof_scores, rnn_target_oof_scores = metrics.regression_scores(
        y_true=targets,
        y_pred=rnn_oof_predictions,
        weights=target_weights,
        target_columns=target_columns
    )
    settings.logger.info(f'rnn OOF Scores: {json.dumps(rnn_global_oof_scores, indent=2)}')

    unet_seresnet_1m_weight = 0.1
    unet_seresnet_4m_weight = 0.3
    unet_seresnet_20m_weight = 0.6
    unet_seresnet_oof_predictions = unet_seresnet_1m_oof_predictions * unet_seresnet_1m_weight + \
                                    unet_seresnet_4m_oof_predictions * unet_seresnet_4m_weight + \
                                    unet_seresnet_20m_oof_predictions * unet_seresnet_20m_weight
    unet_seresnet_oof_predictions[:, hard_target_idx] = unet_seresnet_20m_oof_predictions[:, hard_target_idx]
    unet_seresnet_global_oof_scores, unet_seresnet_target_oof_scores = metrics.regression_scores(
        y_true=targets,
        y_pred=unet_seresnet_oof_predictions,
        weights=target_weights,
        target_columns=target_columns
    )
    settings.logger.info(f'unet_seresnet OOF Scores: {json.dumps(unet_seresnet_global_oof_scores, indent=2)}')

    seq2seq_seresnet_1m_weight = 0.1
    seq2seq_seresnet_4m_weight = 0.3
    seq2seq_seresnet_20m_weight = 0.6
    seq2seq_seresnet_oof_predictions = seq2seq_seresnet_1m_oof_predictions * seq2seq_seresnet_1m_weight + \
                                       seq2seq_seresnet_4m_oof_predictions * seq2seq_seresnet_4m_weight + \
                                       seq2seq_seresnet_20m_oof_predictions * seq2seq_seresnet_20m_weight
    seq2seq_seresnet_oof_predictions[:, hard_target_idx] = seq2seq_seresnet_20m_oof_predictions[:, hard_target_idx]
    seq2seq_seresnet_global_oof_scores, seq2seq_seresnet_target_oof_scores = metrics.regression_scores(
        y_true=targets,
        y_pred=seq2seq_seresnet_oof_predictions,
        weights=target_weights,
        target_columns=target_columns
    )
    settings.logger.info(f'seq2seq_seresnet OOF Scores: {json.dumps(seq2seq_seresnet_global_oof_scores, indent=2)}')

    seq2seq_seresnet_weight = 0.35
    unet_seresnet_weight = 0.35
    rnn_weight = 0.3
    blend_oof_predictions = unet_seresnet_oof_predictions * unet_seresnet_weight + \
                            seq2seq_seresnet_oof_predictions * seq2seq_seresnet_weight + \
                            rnn_oof_predictions * rnn_weight
    blend_global_oof_scores, blend_target_oof_scores = metrics.regression_scores(
        y_true=targets,
        y_pred=blend_oof_predictions,
        weights=target_weights,
        target_columns=target_columns
    )
    settings.logger.info(f'Blend OOF Scores: {json.dumps(blend_global_oof_scores, indent=2)}')

    unet_seresnet_test_predictions = unet_seresnet_1m_test_predictions.copy(deep=True)
    unet_seresnet_test_predictions.iloc[:, 1:] = unet_seresnet_1m_test_predictions.iloc[:, 1:] * unet_seresnet_1m_weight + \
                                                 unet_seresnet_4m_test_predictions.iloc[:, 1:] * unet_seresnet_4m_weight + \
                                                 unet_seresnet_20m_test_predictions.iloc[:, 1:] * unet_seresnet_20m_weight
    unet_seresnet_test_predictions.iloc[:, hard_target_idx + 1] = unet_seresnet_20m_test_predictions.iloc[:, hard_target_idx + 1]

    seq2seq_seresnet_test_predictions = seq2seq_seresnet_1m_test_predictions.copy(deep=True)
    seq2seq_seresnet_test_predictions.iloc[:, 1:] = seq2seq_seresnet_1m_test_predictions.iloc[:, 1:] * seq2seq_seresnet_1m_weight + \
                                                    seq2seq_seresnet_4m_test_predictions.iloc[:, 1:] * seq2seq_seresnet_4m_weight + \
                                                    seq2seq_seresnet_20m_test_predictions.iloc[:, 1:] * seq2seq_seresnet_20m_weight
    seq2seq_seresnet_test_predictions.iloc[:, hard_target_idx + 1] = seq2seq_seresnet_20m_test_predictions.iloc[:, hard_target_idx + 1]

    rnn_test_predictions = gru_10m_test_predictions.copy(deep=True)
    rnn_test_predictions.iloc[:, 1:] = gru_10m_test_predictions.iloc[:, 1:] * gru_10m_weight + \
                                       lstm_10m_test_predictions.iloc[:, 1:] * lstm_10m_weight

    blend_test_predictions = unet_seresnet_test_predictions.copy(deep=True)
    blend_test_predictions.iloc[:, 1:] = unet_seresnet_test_predictions.iloc[:, 1:] * unet_seresnet_weight + \
                                         seq2seq_seresnet_test_predictions.iloc[:, 1:] * seq2seq_seresnet_weight + \
                                         rnn_test_predictions.iloc[:, 1:] * rnn_weight

    q0002_replace_idx = np.array([
        120, 121, 122, 123, 124, 125, 126, 127, 128,
        129, 130, 131, 132, 133, 134, 135, 136, 137,
        138, 139, 140, 141, 142, 143, 144, 145, 146
    ])
    q0002_replace_features = test_features[:, q0002_replace_idx]
    blend_test_predictions.iloc[:, q0002_replace_idx + 1] = -q0002_replace_features / 1200
    blend_test_predictions.iloc[:, 1:] *= target_weights

    blend_test_predictions.to_parquet(settings.MODELS / 'blend' / 'submission.parquet')

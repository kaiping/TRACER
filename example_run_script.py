from trainer import Trainer

BATCH_SIZE = 512
EPOCH_NUM = 500

if __name__ == '__main__':
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings

    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    args = dict()
    args['input_path'] = 'input_dataset.pickle'
    args['output_path'] = 'output_path'

    args['lr'] = 'learning_rate'
    args['weight_decay'] = 'weight_decay'
    args['rnn_dim'] = 'weight_decay'
    args['film_rnn_dim'] = 'film_rnndim'
    args['global'] = True
    args['local'] = True
    args['bidirect'] = True

    model_name = 'LocgloModel'
    model = globals()[model_name]

    trainer = Trainer(args)
    trainer.load_dataset(args['input_path'])
    args['fea_dim'] = trainer.X_train.shape[-1]
    model = model(args)
    trainer.gru_model = model
    trainer.train(
        epoch_num=EPOCH_NUM,
        batch_size=BATCH_SIZE
    )
    trainer.test()

from gomokuZero.pre_train.supervised_learning import Trainer
from gomokuZero.utils import check_load_path

def run(sample_path, json_path,
        weights_path, optimizer_path,
        batch_size, epochs,
        save_path,
        history_path,
        blocks=3, filters=64,
        create_function_name='create_resnet_version_1'):

    if check_load_path(save_path) is None:
        trainer = Trainer(
            sample_path=sample_path,
            json_path=json_path,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            batch_size=batch_size,
            epochs=epochs,
            save_path=save_path,
            history_path=history_path,
            current_epoch=0,
            blocks=blocks,
            filters=filters,
            create_function_name=create_function_name
        )

    else:
        trainer = Trainer.load_trainer(save_path)

    trainer.fit()

if __name__ == '__main__':
    # version = 'yixin_version_'
    # version = 'yixin_version_tf_'
    version = 'tournament_version_tf_'
    # version = 'test_version_'
    # version = 'input_coding_version_'
    # version = 'input_coding_augmentation_version_'

    # version = 'no_augmentation_yixin_version_'

    save_prefix = 'data/pre_train/'
    cache_prefix = 'data/cache/cache_'

    # history_path = 'data/records/yixin_records.npz'
    # sample_path = 'data/records/yixin_samples.npz'

    history_path = 'data/records/tournament_records.npz'
    sample_path = 'data/records/tournament_samples.npz'

    run(
        sample_path=sample_path,
        json_path=save_prefix + version + 'nn_config.json',
        weights_path=save_prefix + version + 'nn_weights.h5',
        optimizer_path=cache_prefix + version + 'optimizer.json',
        batch_size=128,
        epochs=200,
        save_path=cache_prefix + version + 'pre_trainer.json',
        history_path=history_path,
        blocks=3,
        filters=64,
        create_function_name='create_resnet_version_3'
    )

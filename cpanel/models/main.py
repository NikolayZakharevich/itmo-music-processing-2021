from dataset.dumps import save_to_separate_files
from keywords.api import keywords_train_lstm_torch, get_training_data

if __name__ == '__main__':
    # track_ids, labels, label_names = get_training_data()
    #
    # save_to_separate_files(set(track_ids))
    #
    keywords_train_lstm_torch()

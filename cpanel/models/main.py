from app.common.dumps import save_to_separate_files
from app.emotion.api import emotions_train_crnn, get_tracks, TRACKS_TYPE_SINGLE_EMOTION_ONLY, emotions_train_transformer

if __name__ == '__main__':
    # tracks = get_tracks(TRACKS_TYPE_SINGLE_EMOTION_ONLY)
    # save_to_separate_files(set(tracks.ym_track_id))
    emotions_train_crnn()
    # emotions_train_transformer()

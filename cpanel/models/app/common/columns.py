from enum import Enum, unique


@unique
class Column(Enum):
    YM_TRACK_ID = 'ym_track_id'
    YM_ALBUM_ID = 'ym_album_id'
    TRACK_TITLE = 'track_title'
    ARTIST_NAME = 'artist_name'
    COVER_URL = 'cover_url'
    YM_GENRE = 'ym_genre'
    YEAR = 'year'
    EMOTIONS_MAYBE = 'emotions_maybe'
    AUDIO_PATH = 'audio_path'
    LYRICS_PATH = 'lyrics_path'
    COVER_PATH = 'cover_path'
    DEEZER_GENRES = 'deezer_genres'
    ARTIST_COUNTRY = 'artist_country'
    ARTIST_SEX = 'artist_sex'
    ARTIST_YEAR = 'artist_year'
    EMOTIONS = 'emotions'


@unique
class Emotion(Enum):
    COMFORTABLE = 'comfortable'
    HAPPY = 'happy'
    INSPIRATIONAL = 'inspirational'
    JOY = 'joy'
    LONELY = 'lonely'
    FUNNY = 'funny'
    NOSTALGIC = 'nostalgic'
    PASSIONATE = 'passionate'
    QUIET = 'quiet'
    RELAXED = 'relaxed'
    ROMANTIC = 'romantic'
    SADNESS = 'sadness'
    SOULFUL = 'soulful'
    SWEET = 'sweet'
    SERIOUS = 'serious'
    ANGER = 'anger'
    WARY = 'wary'
    SURPRISE = 'surprise'
    FEAR = 'fear'


EMOTIONS = [e.value for e in Emotion]


def map_to_emotion_names(emotions):
    return list(map(lambda e: EMOTIONS[e], emotions))


@unique
class Genre(Enum):
    ROCK = 'rock'
    ALTERNATIVE = 'alternative'
    BLUES = 'blues'
    CLASSICAL = 'classical'
    METAL = 'metal'
    JAZZ = 'jazz'
    COUNTRY = 'country'
    DANCE = 'dance'
    ELECTRONIC = 'electronic'
    POP = 'pop'
    FOLK = 'folk'
    SOUNDTRACK = 'soundtrack'
    BARD = 'bard'
    RAP = 'rap'
    HOUSE = 'house'
    INDIE = 'indie'
    LOUNGE = 'lounge'
    MODERN = 'modern'
    PUNK = 'punk'
    RELAX = 'relax'
    RNB = 'rnb'


GENRES_OLD = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]


def map_genre_raw(genre_raw) -> Genre:
    mapping = {
        'allrock': Genre.ROCK,
        'alternative': Genre.ALTERNATIVE,
        'blues': Genre.BLUES,
        'classical': Genre.CLASSICAL,
        'classicalmasterpieces': Genre.CLASSICAL,
        'classicmetal': Genre.METAL,
        'conjazz': Genre.JAZZ,
        'country': Genre.COUNTRY,
        'dance': Genre.DANCE,
        'disco': Genre.DANCE,
        'dub': Genre.ELECTRONIC,
        'dubstep': Genre.ELECTRONIC,
        'electronics': Genre.ELECTRONIC,
        'epicmetal': Genre.METAL,
        'estrada': Genre.POP,
        'eurofolk': Genre.FOLK,
        'extrememetal': Genre.METAL,
        'films': Genre.SOUNDTRACK,
        'folk': Genre.FOLK,
        'folkmetal': Genre.FOLK,
        'folkrock': Genre.FOLK,
        'foreignbard': Genre.BARD,
        'foreignrap': Genre.RAP,
        'hardcore': Genre.ROCK,
        'hardrock': Genre.ROCK,
        'house': Genre.HOUSE,
        'indie': Genre.INDIE,
        'japanesepop': Genre.POP,
        'jazz': Genre.JAZZ,
        'kpop': Genre.POP,
        'latinfolk': Genre.FOLK,
        'local-indie': Genre.INDIE,
        'lounge': Genre.LOUNGE,
        'meditation': Genre.LOUNGE,
        'metal': Genre.METAL,
        'modern': Genre.MODERN,
        'newage': Genre.ELECTRONIC,
        'newwave': Genre.ROCK,
        'numetal': Genre.ROCK,
        'pop': Genre.POP,
        'posthardcore': Genre.ROCK,
        'postrock': Genre.ROCK,
        'prog': Genre.METAL,
        'progmetal': Genre.METAL,
        'punk': Genre.PUNK,
        'rap': Genre.RAP,
        'relax': Genre.RELAX,
        'rnb': Genre.RNB,
        'rnr': Genre.ROCK,
        'rock': Genre.ROCK,
        'rusbards': Genre.BARD,
        'rusestrada': Genre.POP,
        'rusfolk': Genre.FOLK,
        'ruspop': Genre.POP,
        'rusrap': Genre.RAP,
        'rusrock': Genre.ROCK,
        'ska': Genre.PUNK,
        'soul': Genre.RNB,
        'soundtrack': Genre.SOUNDTRACK,
        'stonerrock': Genre.ROCK,
        'techno': Genre.ELECTRONIC,
        'tradjazz': Genre.JAZZ,
        'turkishpop': Genre.POP,
        'tvseries': Genre.SOUNDTRACK,
        'ukrrock': Genre.ROCK,
        'videogame': Genre.SOUNDTRACK,
    }
    return mapping[genre_raw]

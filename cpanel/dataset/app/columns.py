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
    INDIE = 'indie'
    LOUNGE = 'lounge'
    MODERN = 'modern'
    PUNK = 'punk'
    RNB = 'rnb'
    OTHER = 'other'


@unique
class GenreNational(Enum):
    AZERBAIJANI = 'azerbaijani'
    AMERICAN = 'amerfolk'
    ARGENTINIAN = 'argentinetango'
    ARMENIAN = 'armenian'
    AFRICAN = 'african'
    BALKAN = 'balkan'
    EASTERN = 'eastern'
    GEORGIAN = 'georgian'
    JEWISH = 'jewish'
    EUROPEAN = 'eurofolk'
    CAUCASIAN = 'caucasian'
    CELTIC = 'celtic'
    LATIN = 'latinfolk'
    RUSSIAN = 'rusfolk'


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
        'house': Genre.ELECTRONIC,
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
    return mapping.get(genre_raw, Genre.OTHER)


@unique
class AssignmentColumn(Enum):
    TRACK_ID = Column.YM_TRACK_ID.value
    EMOTION = 'emotion'
    NO = 'NO'
    YES = 'YES'
    RATE = 'rate'

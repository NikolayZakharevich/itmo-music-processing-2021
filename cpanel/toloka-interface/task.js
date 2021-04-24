exports.Task = extend(TolokaHandlebarsTask, function (options) {
  TolokaHandlebarsTask.call(this, options);
}, {
  onRender: function() {
    const root = this.getDOMElement();
    const el = root.querySelector('#emotion-words');
    const emotion = el.innerHTML
    el.innerHTML = getEmotionWords(emotion) + ' (' + emotion + ')';

    // DOM-элемент задания сформирован (доступен через #getDOMElement()
  },
  onDestroy: function() {
    // Задание завершено, можно освобождать (если были использованы) глобальные ресурсы
  }
});

function getEmotionWords(emotion) {
  switch (emotion) {
    case 'comfortable':
      return 'Комфортная, уютная, домашняя';
    case 'happy':
      return 'Счастливая';
    case 'inspirational':
      return 'Вдохновляющая';
    case 'joy':
      return 'Радостная, позитивная, бодрая';
    case 'lonely':
      return 'Одинокая';
    case 'funny':
      return 'Забавная, смешная';
    case 'nostalgic':
      return 'Ностальгическая';
    case 'passionate':
      return 'Страстная, сексуальная';
    case 'quiet':
      return 'Спокойная, тихая';
    case 'relaxed':
      return 'Расслабленная, релакс';
    case 'romantic':
      return 'Романтическая';
    case 'sadness':
      return 'Печальная, грустная';
    case 'soulful':
      return 'Душевная';
    case 'sweet':
      return 'Сладкая, милая';
    case 'serious':
      return 'Серьёзная';
    case 'anger':
      return 'Ярость, гнев';
    case 'wary':
      return 'Острожная, напряжённая';
    case 'surprise':
      return 'Удивительная, сюрприз';
    case 'fear':
      return 'Страх';

    default:
      return emotion;
  }
}

function extend(ParentClass, constructorFunction, prototypeHash) {
  constructorFunction = constructorFunction || function () {};
  prototypeHash = prototypeHash || {};
  if (ParentClass) {
    constructorFunction.prototype = Object.create(ParentClass.prototype);
  }
  for (var i in prototypeHash) {
    constructorFunction.prototype[i] = prototypeHash[i];
  }
  return constructorFunction;
}

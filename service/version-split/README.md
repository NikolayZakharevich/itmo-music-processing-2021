# Сервисы для работы с музыкой

Примеры развернутых приложений:

- [Сервис для предсказания эмоций](https://sky4uk.xyz/demo/music-emotions/)
- [Сервис для предсказания шрифта](https://sky4uk.xyz/demo/music-fonts/)
- [Сервис для предсказания ключевых слов](https://sky4uk.xyz/demo/music-keywords/)

## Сервис для предсказания эмоций

Находится в папке `emotions`

Настройки задаются в
файле [config.yml](https://github.com/NikolayZakharevich/music-keywords/blob/master/emotions/config.yml)
, по умолчанию разворачивается на порту `8080`, название метода — `music-emotions` (то есть сервис будет доступен по
адресу `0.0.0.0:8081/demo/music-emotions`):

```yaml
app:
  port: 8080
  host: "0.0.0.0"
  thread_pool: 5
  tmp_dir: "tmp/"
  method_name: "music-emotions"
```

Скрипт для развертывания:

```sh
cd emotions
docker build -t "music-emotions:Dockerfile" .
docker run --publish 8081:8080 music-emotions:Dockerfile
```

Пример развернутого приложения: <https://sky4uk.xyz/demo/music-emotions/>

## Сервис для предсказания шрифта

Находится в папке `fonts`

Настройки задаются в
файле [config.yml](https://github.com/NikolayZakharevich/music-keywords/blob/master/fonts/config.yml)
, по умолчанию разворачивается на порту `8080`, название метода — `music-fonts` (то есть сервис будет доступен по
адресу `0.0.0.0:8082/demo/music-fonts`):

```yaml
app:
  port: 8080
  host: "0.0.0.0"
  thread_pool: 5
  tmp_dir: "tmp/"
  method_name: "music-fonts"
```

Скрипт для развертывания:

```sh
cd fonts
docker build -t "music-fonts:Dockerfile" .
docker run --publish 8082:8080 music-fonts:Dockerfile
```

Пример развернутого приложения: <https://sky4uk.xyz/demo/music-fonts/>

## Сервис для предсказания ключевых слов

Находится в папке `keywords` 

Настройки задаются в
файле [config.yml](https://github.com/NikolayZakharevich/music-keywords/blob/master/keywords/config.yml)
, по умолчанию разворачивается на порту `8080`, название метода — `music-keywords` (то есть сервис будет доступен по
адресу `0.0.0.0:8083/demo/music-keywords`):

```yaml
app:
  port: 8080
  host: "0.0.0.0"
  thread_pool: 5
  tmp_dir: "tmp/"
  method_name: "music-keywords"
```

Скрипт для развертывания:

```sh
cd keywords
docker build -t "music-keywords:Dockerfile" .
docker run --publish 8083:8080 music-keywords:Dockerfile
```

Пример развернутого приложения: <https://sky4uk.xyz/demo/music-keywords/>

## Настройка nginx

Можно делать обычное проксирование на порт, на котором запущено приложение

```nginx
location /demo/music-emotions {
    proxy_pass http://127.0.0.1:8081;
}
location /demo/music-fonts{
    proxy_pass http://127.0.0.1:8082;
}
location /demo/music-keywords {
    proxy_pass http://127.0.0.1:8083;
}
```

**NOTE:** если указать `location /demo/music-emotions/` (то есть со слэшем в конце), то POST-запросы будут превращаться
в GET, ничего не будет работать 
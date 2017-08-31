* 安装要求：
```
tensorflow 1.0+
python 2+
all platform
```

* 安装：
```
git clone https://github.com/jinfagang/tensorflow_poems.git
```

* 使用方法：
```
# for poem train
python main.py -w poem --train
# for lyric train
python main.py -w lyric --train

# for generate poem
python main.py -w poem --no-train
# for generate lyric
python main.py -w lyric --no-train

```

* 参数说明
`-w or --write`: 设置作诗还是创作歌词，poem表示诗，lyric表示歌词
`--train`: 训练标识位，首次运行请先train一下...
`--no-train`: 生成标识位

# Local Chat

## about
This repository aims to reproduce an LLM chat experience.
The project uses llama.cpp to load the models and gpt4all to generate the embedding models.


## How to run
- Install dependencies
```sh
pip install -r requirements.txt
```

- Add your model config file. See `models/` directory for some examples.

- Create a folder inside the documents folder and add your documents into it.

- Run the ingest script
```sh
python localchat/ingest --model models/YOUR_MODEL.json
```

- Run script
```sh
python localchat --model models/YOUR_MODEL.json
```

## How to contribute
Feel free to add an issue or make a pull request.

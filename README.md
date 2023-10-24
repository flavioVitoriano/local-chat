# Local Chat

## about
This repository aims to reproduce an LLM chat experience.
The project uses llama.cpp to load the models and gpt4all to load the embedding models.


## How to run
- First, you need to download a .gguf model, download and put inside bin folder.

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


## Next steps
See `MILESTONES.md` for the next aimed steps to be done.

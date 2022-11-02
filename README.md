# soft-vc-realtime

    A wrapper for Soft-VC to make a real-time voice converter.

----

### Quick Start

- install PyTorch following the official guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), only need **torch** and **torchaudio** parts
- install dependecies `pip install -r requirements.txt`
- run the wiring test `python app.py --mode echo` to check your audio device settings ok
- now, run in default vc mode `python app.py`

#### trouble shooting

Q: When running in vc mode, there's no GUI windows and it directly quits after model loading
A: Open and edit file `%USERPROFILE%\.cache\torch\hub\bshall_hifigan_main\hifigan\utils.py`, comment the 4-th line `matplotlib.use("Agg")`

----

by Armit
2022/11/02

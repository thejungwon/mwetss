import librosa
import requests
import json
from multiprocessing.pool import ThreadPool

def fetch_url(data):
    r = requests.post(data[0],json=data[1])
    return r.text

def get_wer(hp, audio, est_mags, target_path, mixed_paths):
    est_list = []

    for i, mixed_path in enumerate(mixed_paths) :
        wav, _ = librosa.load(mixed_path, hp.audio.sample_rate)
        _, mixed_phase = audio.wav2spec(wav)
        # print(est_mag)
        est_mag = est_mags[i].cpu().detach().numpy()

        est_wav = audio.spec2wav(est_mag, mixed_phase)
        #target_path
        new_path = mixed_path.split("-")[0]+"-est.wav"
        est_list.append(new_path)
        # print(new_path)
        librosa.output.write_wav(new_path, est_wav, hp.audio.sample_rate)
    urls = []
    for i in range(8):
        audio_file1=est_list[i]
        # audio_file2=est_list[i*2+1]
        audio_files = [audio_file1]

        app_json = {'audio_files':audio_files}
        urls.append(["http://localhost:808"+str(i)+"/", app_json])
    pool = ThreadPool(8)
    results = pool.map(fetch_url, urls)

    output={"wers":[],"texts_gt":[],"texts_gen":[]}
    # print(results)
    for result in results:

        result = json.loads(result)

        for val in result['texts_gen']:
                output['texts_gen'].append(val)
        for val in result['texts_gt']:
                output['texts_gt'].append(val)
        for val in result['wers']:
                output['wers'].append(val)

    pool.close()
    pool.join()

    return output['wers']

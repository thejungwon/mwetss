import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import librosa
import requests
import json
import numpy as np
def custom_loss(s1,s2):
    alpha = torch.tensor(0.113).cuda()
    return torch.sum(torch.pow(torch.abs(s2)-torch.abs(s1),2))+alpha*torch.sum(torch.pow(s2-s1,2))
def wmse(t1, t2, wer=[]):
    diff = t1 - t2

    # m_wer = np.median(wer)
    # mse = torch.sum(diff * diff) / diff.numel()
    alpha = 0.3
    # wer = np.array(wer).reshape(diff.shape[0],1,1)
    weight = torch.tensor(alpha+wer)
    weight = weight.type(torch.FloatTensor)
    weight = weight.cuda()

    weighted_mse = torch.sum(weight* diff * diff) / diff.numel()
    # print("normal mse", mse)
    # print("weighted mse", weighted_mse)
    return weighted_mse
def validate(audio,hp, model, embedder, testloader, writer, step, logger):
    model.eval()

    criterion = wmse
    # criterion = nn.MSELoss()
    first = True
    with torch.no_grad():
        sdr_list = []
        test_losses = []
        wer_list = []
        for i, batch in enumerate(testloader):

            for each_batch in batch:

                dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase, target_path, mixed_path = each_batch

                dvec_mel = dvec_mel.cuda()
                target_mag = target_mag.unsqueeze(0).cuda()
                mixed_mag = mixed_mag.unsqueeze(0).cuda()
                target_mag_cuda = target_mag
                mixed_mag_cuda = mixed_mag

                dvec = embedder(dvec_mel)
                dvec = dvec.unsqueeze(0)
                est_mask = model(mixed_mag, dvec)
                est_mag = est_mask * mixed_mag
                est_mag_cuda = est_mag

                # est_mag = torch.pow(torch.clamp(output, min=0.0), hp.audio.power)
                # target_mag = torch.pow(torch.clamp(target_mag, min=0.0), hp.audio.power)


                mixed_mag = mixed_mag[0].cpu().detach().numpy()
                target_mag = target_mag[0].cpu().detach().numpy()
                est_mag = est_mag[0].cpu().detach().numpy()
                est_wav = audio.spec2wav(est_mag, mixed_phase)
                est_mask = est_mask[0].cpu().detach().numpy()
                L = min(target_wav.shape[0], est_wav.shape[0], 16000*3)
                sdr = bss_eval_sources(target_wav[:L], est_wav[:L], False)[0][0]
                sdr_before = bss_eval_sources(target_wav[:L], mixed_wav[:L], False)[0][0]
                sdr_list.append(sdr)


                new_path = mixed_path.split("-")[0]+"-validation.wav"
                librosa.output.write_wav(new_path, est_wav,  hp.audio.sample_rate)
                est_list = [new_path]
                r = requests.post("http://localhost:8088", json={'audio_files':est_list})
                result = json.loads(r.text)
                wer = result['wers'][0]
                logger.info("SDR BEFORE %f, SDR AFTER : %f, WER: %f" % (sdr_before,sdr,wer))
                wer_list.append(wer)
                test_loss = criterion(target_mag_cuda, est_mag_cuda, wer).item()
                # test_loss = criterion(target_mag_cuda, est_mag_cuda).item()
                test_losses.append(test_loss)

                text_gt = result['texts_gt'][0]
                text_gen = result['texts_gen'][0]
            if i == 9:
                break


        logger.info("SDR %f, WER: %f" % (np.mean(sdr_list), np.mean(wer_list)))
        writer.log_evaluation(np.mean(test_losses), np.mean(sdr_list),
                                mixed_wav, target_wav, est_wav,
                                mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                step, np.mean(wer_list), text_gt, text_gen)


    model.train()

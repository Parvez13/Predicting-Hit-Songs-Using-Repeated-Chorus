import librosa
import numpy as np
from scipy.stats import skew, kurtosis


def statistics(list, feature, columns_name, data):
    i = 0
    for ele in list:
        _skew = skew(ele)
        columns_name.append(f'{feature}_kew_{i}')
        min = np.min(ele)
        columns_name.append(f'{feature}_min_{i}')
        max = np.max(ele)
        columns_name.append(f'{feature}_max_{i}')
        std = np.std(ele)
        columns_name.append(f'{feature}_std_{i}')
        mean = np.mean(ele)
        columns_name.append(f'{feature}_mean_{i}')
        median = np.median(ele)
        columns_name.append(f'{feature}_median_{i}')
        _kurtosis = kurtosis(ele)
        columns_name.append(f'{feature}_kurtosis_{i}')

        i += 1
        data.append(_skew)
        data.append(min)
        data.append(max)
        data.append(std)
        data.append(mean)
        data.append(median)
        data.append(_kurtosis)

    return data


def extract_features(audio_path, title, artist_name):

    data = []
    columns_name = ['Artist', 'Title']
    data.append(artist_name)
    data.append(title)


    x , sr = librosa.load(audio_path)

    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
    statistics(chroma_stft, 'chroma_stft', columns_name, data)

    chroma_cqt = librosa.feature.chroma_cqt(y=x, sr=sr)
    statistics(chroma_cqt, 'chroma_cqt', columns_name, data)

    chroma_cens = librosa.feature.chroma_cens(y=x, sr=sr)
    statistics(chroma_cens, 'chroma_cens', columns_name, data)

    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    statistics(mfcc, 'mfcc', columns_name, data)

    rms = librosa.feature.rms(y=x)
    statistics(rms, 'rms', columns_name, data)

    spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
    statistics(spectral_centroid, 'spectral_centroid', columns_name, data)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    statistics(spectral_bandwidth, 'spectral_bandwidth', columns_name, data)

    spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
    statistics(spectral_contrast, 'spectral_contrast', columns_name, data)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    statistics(spectral_rolloff, 'spectral_rolloff', columns_name, data)

    tonnetz = librosa.feature.tonnetz(y=x, sr=sr)
    statistics(tonnetz, 'tonnetz', columns_name, data)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x)
    statistics(zero_crossing_rate, 'zero_crossing_rate', columns_name, data)

    return data, columns_name

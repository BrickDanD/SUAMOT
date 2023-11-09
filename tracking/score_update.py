def score_update(tracks, detections, method):
    if method == 'add':
        score = tracks[10] + detections[7]
    elif method == 'max':
        score = max(tracks[10], detections[7])
    elif method == 'multiplication':
        score = 1 - (1 - tracks[10]) * (1 - detections[7])
    else:
        score = 1 - (1 - tracks[10]) * (1 - detections[7]) / ((1 - tracks[10]) + (1 - detections[7]))

    return score

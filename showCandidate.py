from vis import showCandidate,findPositiveSamples

positiveSamples = findPositiveSamples(limit=1)
uid = positiveSamples[0].series_uid
print(len(positiveSamples))

showCandidate(series_uid = uid)


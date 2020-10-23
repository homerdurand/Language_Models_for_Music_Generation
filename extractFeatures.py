from os import listdir
from os.path import isfile, join
import numpy as np
import music21 as m21
from tqdm import tqdm

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def transposeMidi(midi, key="C") :
    k = midi.analyze('key')
    i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(key))
    midiNew = midi.transpose(i)
    return midiNew

def arrayFromListofList(listOfList, integer=False):
    x = listOfList
    length = max(map(len, x))
    if integer :
        y=np.array([xi+[0]*(length-len(xi)) for xi in x])
        for i in range(len(y)) :
            for j in range(1,len(y[i])) :
                if y[i][j] == 0 :
                    y[i][j] = y[i][j-1]
    else :
        y=np.array([xi+[0.0]*(length-len(xi)) for xi in x])
    return y

def getMelodySteps(melody) :
    steps = np.array(melody[0])
    for i in range(1,len(melody)) :
        if melody[i]=="R":
            steps = np.append(steps,0)
        else :
            steps = np.append(steps,melody[i]-melody[i-1])
    return steps

def steps2melody(steps) :
    melody = [steps[0]]
    for i in range(1, len(steps)) :
        melody.append(melody[-1]+steps[i])
    return melody

def getMelodyFeatures(midiPath) :
    midiMelody = m21.converter.parse(midiPath).parts[0]
    notes = []
    tempo = []
    for element in midiMelody :
        if isinstance(element,m21.note.Note) :
            notes.append(element.pitch.midi)
            tempo.append(element.duration.quarterLength)
        elif isinstance(element, m21.note.Rest) :
            notes.append(0)
            tempo.append(element.duration.quarterLength)
    return notes, tempo

def chorale2melody(midiPath, transpose = True) :
    midiTemp= m21.converter.parse(midiPath)
    if transpose :
        midiTemp = transposeMidi(midiTemp, "C")
    midiParts = midiTemp.parts

    nbParts = len(midiParts)
    notes = []
    for i in range(nbParts) :
        for element in midiParts[i] :
            if isinstance(element,m21.note.Note) :
                notes.append(element.pitch.name + str(element.octave) + "°" + str(element.duration.quarterLength))
            elif isinstance(element, m21.note.Rest) :
                notes.append("R°" + str(element.duration.quarterLength))
    return notes


def getChoralFeatures(midiPath) :
    midiParts = m21.converter.parse(midiPath).parts
    notesMatrix = []
    tempoMatrix = []
    for i in range(len(midiParts)) :
        midiMelody = midiParts[i]
        notes = []
        tempo = []
        for i in range(len(midiMelody)) :
            if isinstance(midiMelody[i],m21.note.Note) :
                notes.append(midiMelody[i].pitch.midi)
                tempo.append(midiMelody[i].duration.quarterLength)
            elif isinstance(midiMelody[i], m21.note.Rest) and (isinstance(midiMelody[i-1], m21.note.Note) or isinstance(midiMelody[i-1], m21.note.Rest)) :
                notes.append(midiMelody[i-1].pitch.midi)
                tempo.append(midiMelody[i].duration.quarterLength)
        notesMatrix.append(notes)
        tempoMatrix.append(tempo)
    return arrayFromListofList(notesMatrix, integer = True), arrayFromListofList(tempoMatrix)

def features2midi(notesMatrix, tempoMatrix) :
    score = m21.stream.Score()
    for voice in range(len(notesMatrix)) :
        stream = m21.stream.Stream()
        for i in range(len(notesMatrix[0])) :
            duration = m21.duration.Duration(tempoMatrix[voice][i])
            note = m21.note.Note(notesMatrix[voice][i], duration = duration)
            stream.append(note)
        score.insert(0,stream)
    return score

def melody2stream(melody) :
    stream = m21.stream.Stream()
    for element in  melody:
        if element == "</s>" :
            return stream
        elif element != "<s>" :
            noteStr, durationStr = element.split("°")
            duration = m21.duration.Duration(convert_to_float(durationStr))
            if noteStr == "R" :
                note = m21.note.Rest(duration = duration)
            else :
                note = m21.note.Note(noteStr, duration = duration)
            stream.append(note)
    return stream

def createDataBase(directory) :
    fichiers = [f for f in listdir(directory) if isfile(join(directory, f))]
    data = []
    for fichier in tqdm(fichiers) :
        try :
            midiPath=directory+fichier
            encodedChorale = getChoralFeatures(midiPath)
            data.append(encodedChorale)
        except Exception as e:
            pass

    return data

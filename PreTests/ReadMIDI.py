from music21 import converter, chord, note

dataset_name = "clean_midi"
artist_name = "Bach Johann Sebastian"
song_name = "Prelude in G"
file = "../data/{}/{}/{}.mid".format(dataset_name, artist_name, song_name)

score = converter.parse(file).chordify()

notes = []
durations = []

for element in score.flat:
    if isinstance(element, chord.Chord):
        notes.append(".".join(n.nameWithOctave for n in element.pitches))
        durations.append(element.duration.quarterLength)

    if isinstance(element, note.Note):
        if element.isRest:
            notes.append(str(element.name))
        else:
            notes.append(str(element.nameWithOctave))

        durations.append(element.duration.quarterLength)


print(notes)
print(durations)

for x, y in zip(notes, durations):
    print(x, y)

print(score.analyze("key"))

from music21 import converter, chord, note, stream
import glob
import itertools
from collections import namedtuple, defaultdict
from Utils.Util import SongMidi
from tqdm import tqdm
from pathlib import PurePath
import pickle

limit_data = None

REST_VALUE = "rest"

tone_values = {
    "C-": 11, "C": 0, "C#": 1,
    "D-": 1, "D": 2, "D#": 3,
    "E-": 3, "E": 4, "E#": 5,
    "F-": 4, "F": 5, "F#": 6,
    "G-": 6, "G": 7, "G#": 8,
    "A-": 8, "A": 9, "A#": 10,
    "B-": 10, "B": 11, "B#": 0
}

tone_values_reverse = {
    0: "C", 1: "C#",
    2: "D", 3: "D#",
    4: "E",
    5: "F", 6: "F#",
    7: "G", 8: "G#",
    9: "A", 10: "A#",
    11: "B"
}


def is_rest(tone):
    return tone == REST_VALUE


def get_tone_from_value(tonal_value):
    floor_val, mod = divmod(tonal_value, 12)
    return tone_values_reverse[mod] + str(floor_val)


def get_tonal_value(tone):
    if len(tone) == 3:
        return tone_values[tone[:-1]] + int(tone[-1]) * 12
    elif len(tone) == 1 or (tone[1] == "-" or tone[1] == "#"):
        return tone_values[tone]
    else:
        return tone_values[tone[:-1]] + int(tone[-1]) * 12


def parse_score(part):
    notes = []
    durations = []

    for element in part.flat.notesAndRests:
        if isinstance(element, chord.Chord):
            notes.append(".".join(n.nameWithOctave for n in element.pitches))
            durations.append(element.duration.quarterLength)
        elif isinstance(element, note.Note):
            notes.append(str(element.nameWithOctave))
            durations.append(element.duration.quarterLength)
        elif isinstance(element, note.Rest):
            notes.append(str(element.name))
            durations.append(element.duration.quarterLength)

    return notes, durations


def parse_midi_part(part, lowest_relative_note_steps=True, octave_break=2, octave_limit=3):
    n, d = parse_score(part)
    all_notes = list(itertools.chain.from_iterable(
        [element.split(".") for element in n if not is_rest(element)]))
    lowest_note = min(all_notes, key=lambda element: get_tonal_value(element))

    for i in range(len(n)):
        if not is_rest(n[i]):
            n[i] = [get_tonal_value(subnote) for subnote in n[i].split(".")]
        else:
            n[i] = REST_VALUE

    tracks = []
    durations = [d.copy()]

    lowest_notes = [lowest_note]

    octave_break_val = octave_break * 12
    octave_limit_val = octave_limit * 12
    offset = get_tonal_value(lowest_note)
    highest_note = max(all_notes, key=lambda element: get_tonal_value(element))
    high_value = get_tonal_value(highest_note)
    note_range = high_value - offset

    if isinstance(octave_break, int) and octave_break < 10 and note_range > octave_break_val:
        break_point = note_range // 2 + offset

        track = []
        bass_track = []

        lowest_note_track = high_value

        for element in n:
            if is_rest(element):
                track.append(element)
                bass_track.append(element)
            else:
                track_note = []
                bass_note = []
                for note_element in element:
                    if note_element >= break_point:
                        track_note.append(note_element)

                        if lowest_note_track > note_element:
                            lowest_note_track = note_element
                    else:
                        bass_note.append(note_element)

                if len(track_note) == 0:
                    track.append(REST_VALUE)
                else:
                    track.append(track_note)

                if len(bass_note) == 0:
                    bass_note.append(REST_VALUE)
                else:
                    bass_note.append(bass_note)

        durations.append(d.copy())
        lowest_notes.append(get_tone_from_value(lowest_note_track))
        tracks.append(bass_track)
        tracks.append(track)
    else:
        tracks.append(n)

    if lowest_relative_note_steps:
        for low, track in zip(lowest_notes, tracks):
            offset_val = get_tonal_value(low)
            for note_list in track:
                if not is_rest(note_list):
                    for i in range(len(note_list)):
                        note_list[i] -= offset_val

    if octave_limit is not None:
        for track in tracks:
            for i in range(len(track)):
                if not is_rest(track[i]):
                    track[i] = [val % octave_limit_val for val in track[i]]

    for track in tracks:
        for i in range(len(track)):
            if not is_rest(track[i]):
                track[i] = sorted(list(set(track[i])))
                track[i] = ".".join(map(str, track[i]))

    return lowest_notes, tracks, durations


def parse_midi_file(filename, lowest_relative_note_steps=True,
                    octave_break=2, octave_limit=3):
    song = converter.parse(filename)
    notes = []
    durations = []
    lowest_notes = []

    for part in song.parts:
        lowest_note, n, d = parse_midi_part(part, lowest_relative_note_steps,
                                            octave_break, octave_limit)
        notes.extend(n)
        durations.extend(d)
        lowest_notes.extend(lowest_note)
    splitted = PurePath(filename.replace(".mid", "")).parts
    return SongMidi(splitted[-1], splitted[-2], splitted[-4], splitted[-3],
                    lowest_notes, notes, durations)


def parse_all_midi_files(root_path, limit_read=None, lowest_relative_note_steps=True,
                         octave_break=2, octave_limit=3):
    songs = []
    files = list(glob.iglob(root_path + '**/*.mid', recursive=True))

    if isinstance(limit_read, int) and 0 < limit_read < len(files):
        files = files[:limit_read]

    files = tqdm(files)

    for filename in files:
        try:
            songs.append(parse_midi_file(filename, lowest_relative_note_steps, octave_break, octave_limit))
        except BaseException as err:
            print(err)

    return songs


midi_data = parse_all_midi_files("../data/adl-piano-midi/", limit_data, True, 2, 2)

note_variations = defaultdict(int)
duration_variations = defaultdict(int)

for song in midi_data:
    for durations in song.durations:
        for duration in durations:
            duration_variations[duration] += 1
    for notes in song.notes:
        for note in notes:
            note_variations[note] += 1

for key in note_variations:
    print(key, note_variations[key])

print(len(note_variations), len(duration_variations), len(max(midi_data, key=lambda data: len(data.notes))))

with open("../midi_data.dump", "wb") as file:
    pickle.dump(midi_data, file)

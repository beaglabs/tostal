import random
import math
from typing import Iterator


LITHOLOGY_NAMES = [
    "shale", "siltstone", "sandstone", "conglomerate",
    "limestone", "dolostone", "marl", "evaporite",
    "coal", "chert", "basalt", "granite",
]

WELL_NAMES = [
    "B-17", "C-03", "A-12", "D-05", "E-22", "F-09",
    "G-14", "H-01", "I-08", "J-15", "K-04", "L-11",
]

TEMPLATES = [
    "Well {well} contains {n_units} distinct lithological units. {unit_descriptions}.",
    "Analysis of well {well} reveals {n_units} major units. {unit_descriptions}",
    "The {well} well shows {n_units} lithological zones. {unit_descriptions}",
    "In well {well}, {n_units} units were identified: {unit_list}.",
    "Wireline logs from {well} indicate {n_units} formations. {unit_descriptions}",
    "Based on log analysis, well {well} penetrates {n_units} distinct intervals. {unit_descriptions}",
    "Subsurface mapping at {well} resolves {n_units} layers. {unit_descriptions}",
    "Well {well} exhibits {n_units} depositional sequences. {unit_descriptions}",
]

TRANSITION_WORDS = [
    "The uppermost unit", "Below this,", "The lowermost unit", "A transitional",
    "The middle", "Underlying this,", "This is overlain by", "Beneath the",
    "The deepest interval", "Starting at the surface,",
]

UNCERTAINTY_PHRASES = [
    "",
    " This classification has {conf}% confidence.",
    " Confidence: {conf}%.",
    " (certainty: {conf}%)",
]

DETAIL_PHRASES = [
    "",
    " Gamma ray values average {gr} API with resistivity around {res} ohm-m.",
    " Log character suggests {gr} API gamma and {res} ohm-m resistivity.",
    " The gamma ray is {gr} API on average.",
    " Resistivity measures approximately {res} ohm-m.",
]


def generate_geology_text(num_examples: int, seq_len: int = 96, seed: int = 42) -> Iterator[str]:
    random.seed(seed)
    generated = 0
    while generated < num_examples:
        well = random.choice(WELL_NAMES)
        n_units = random.randint(2, 4)
        units = random.sample(LITHOLOGY_NAMES, n_units)

        depths = []
        cursor = 0
        for i in range(n_units):
            thickness = random.randint(100, 800)
            depths.append((cursor, cursor + thickness))
            cursor += thickness

        template = random.choice(TEMPLATES)

        descs = []
        for i, (unit, (d_start, d_end)) in enumerate(zip(units, depths)):
            conf = random.randint(60, 98)
            gr = random.randint(15, 150)
            res = random.randint(2, 60)
            unc = random.choice(UNCERTAINTY_PHRASES).format(conf=conf)
            det = random.choice(DETAIL_PHRASES).format(gr=gr, res=res)

            if i == 0:
                prefix = random.choice([TRANSITION_WORDS[0], ""])
            else:
                prefix = random.choice(TRANSITION_WORDS[1:])

            desc = f"{prefix} {unit} from {d_start} to {d_end}m{unc}{det}"
            descs.append(desc.strip())

        unit_descriptions = ". ".join(descs) + "."
        unit_list = ", ".join(units)

        text = template.format(
            well=well, n_units=n_units, unit_descriptions=unit_descriptions, unit_list=unit_list
        )

        generated += 1
        yield text[:seq_len * 8]


def geology_text_generator(vocab_size: int, seq_len: int, num_batches: int, seed: int = 42):
    texts = generate_geology_text(num_batches, seq_len, seed)
    import torch
    for text in texts:
        ids = [ord(c) % (vocab_size - 1) + 1 for c in text[:seq_len]]
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))
        yield torch.tensor(ids, dtype=torch.long).unsqueeze(0)
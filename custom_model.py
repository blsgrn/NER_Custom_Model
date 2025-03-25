import spacy
import random
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags
from spacy.util import minibatch

# Load the pre-trained SciSpaCy model
nlp = spacy.load("en_core_sci_md")

# Define custom labels (e.g., Drug, Dosage, etc.)
CUSTOM_LABELS = ["DRUG", "DOSAGE", "FREQUENCY"]

# Get the Named Entity Recognition (NER) component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add custom labels to the NER component
for label in CUSTOM_LABELS:
    ner.add_label(label)

# Prepare training data
TRAINING_DATA = [
    ("Prescription for Paracetamol 500mg, take twice daily",
     {"entities": [(19, 28, "DRUG"), (29, 34, "DOSAGE"), (35, 49, "FREQUENCY")]}),

    ("Aspirin 100mg should be taken once a day.",
     {"entities": [(0, 7, "DRUG"), (8, 13, "DOSAGE"), (34, 45, "FREQUENCY")]}),

    ("Ibuprofen 200mg, take every 6 hours.",
     {"entities": [(0, 9, "DRUG"), (10, 15, "DOSAGE"), (22, 33, "FREQUENCY")]}),

    ("Take Amoxicillin 250mg three times a day.",
     {"entities": [(5, 16, "DRUG"), (17, 22, "DOSAGE"), (23, 41, "FREQUENCY")]}),

    ("Metformin 500mg should be taken twice a day.",
     {"entities": [(0, 9, "DRUG"), (10, 15, "DOSAGE"), (35, 46, "FREQUENCY")]}),

    ("You should take Warfarin 3mg every evening.",
     {"entities": [(17, 25, "DRUG"), (26, 29, "DOSAGE"), (30, 43, "FREQUENCY")]}),

    ("Clarithromycin 500mg, take once every 12 hours.",
     {"entities": [(0, 14, "DRUG"), (15, 20, "DOSAGE"), (27, 42, "FREQUENCY")]}),

    ("Apply Neosporin ointment to the wound twice daily.",
     {"entities": [(6, 15, "DRUG"), (38, 50, "FREQUENCY")]}),  # No DOSAGE here

    ("Prednisone 10mg to be taken every morning.",
     {"entities": [(0, 9, "DRUG"), (10, 14, "DOSAGE"), (28, 41, "FREQUENCY")]}),

    ("Losartan 50mg once per day to control blood pressure.",
     {"entities": [(0, 8, "DRUG"), (9, 14, "DOSAGE"), (15, 26, "FREQUENCY")]}),

    ("Take Atorvastatin 20mg at night before bed.",
     {"entities": [(5, 17, "DRUG"), (18, 22, "DOSAGE"), (23, 41, "FREQUENCY")]}),

    ("Simvastatin 40mg should be taken before sleeping.",
     {"entities": [(0, 11, "DRUG"), (12, 16, "DOSAGE"), (34, 49, "FREQUENCY")]}),

    ("Gabapentin 300mg is prescribed for three times daily.",
     {"entities": [(0, 9, "DRUG"), (10, 16, "DOSAGE"), (31, 47, "FREQUENCY")]}),

    ("Hydrochlorothiazide 25mg should be taken in the morning.",
     {"entities": [(0, 18, "DRUG"), (19, 23, "DOSAGE"), (43, 55, "FREQUENCY")]}),

    ("Take Diazepam 5mg when needed for anxiety.",
     {"entities": [(5, 13, "DRUG"), (14, 17, "DOSAGE")]}),  # No fixed frequency

    ("Omeprazole 20mg is taken every morning before breakfast.",
     {"entities": [(0, 9, "DRUG"), (10, 14, "DOSAGE"), (24, 38, "FREQUENCY")]}),

    ("Levothyroxine 75mcg, take on an empty stomach every day.",
     {"entities": [(0, 12, "DRUG"), (13, 18, "DOSAGE"), (42, 50, "FREQUENCY")]}),

    ("Use Salbutamol inhaler when necessary.",
     {"entities": [(4, 13, "DRUG")]}),  # No DOSAGE or FREQUENCY

    ("Insulin 10 units before meals daily.",
     {"entities": [(0, 7, "DRUG"), (8, 16, "DOSAGE"), (30, 35, "FREQUENCY")]}),

    ("Take Codeine 15mg as needed for pain.",
     {"entities": [(5, 12, "DRUG"), (13, 17, "DOSAGE")]}),  # No fixed frequency
]


# Disable other pipes except 'ner' for efficiency
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()

    # Training loop
    for i in range(30):  # Reduce iterations for efficiency
        random.shuffle(TRAINING_DATA)
        losses = {}
        batches = minibatch(TRAINING_DATA, size=2)
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Iteration {i + 1}, Losses: {losses}")

# Save the fine-tuned model
nlp.to_disk("/home/balasegaran/PycharmProjects/Custom_NLP_Models/custom_medical_model")

print("Fine-tuned model saved successfully!")

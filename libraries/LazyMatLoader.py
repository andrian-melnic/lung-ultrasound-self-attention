import scipy.io as sio

class LazyMatLoader:
    """Lazy-loading for .mat files."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load(self):
        if self.data is None:
            self.data = sio.loadmat(self.file_path)

    def get_video_name(self):
        self.load()
        return self.file_path.stem

    def get_frame(self, frame_idx):
        self.load()
        frames = self.data['frames']
        return frames[:, :, :, frame_idx]

    def get_score(self, frame_idx):
        self.load()
        scores = self.data['Score_matrix']
        score = [scores[0][frame_idx], scores[1][frame_idx], scores[2][frame_idx]]
        # scores[x, y] --> x = 3 ; y = n. frames del video .mat corrispondente
        return score

    def get_num_frames(self):
        self.load()
        frames = self.data.get('frames', None)
        retval = -1
        if frames is not None and len(frames.shape) > 3:
            retval = frames.shape[3]
        return retval

    def get_patient(self):
        self.load()
        patient_reference = self.file_path.parent.name
        return patient_reference

    def get_medical_center(self):
        self.load()
        medical_center = self.file_path.parent.parent.name

        # Mappa dei pazienti ai centri medici corretti
        patient_to_medical_center = {
            "Paziente 1": "Lucca (NCD)",
            "Paziente 2": "Lucca (NCD)",
            "Paziente 3": "Lucca (NCD)",
            "Paziente 4": "Lucca (NCD)",
            "Paziente 5": "Lucca (NCD)",
            "Paziente 6": "Lucca (NCD)",
            "Paziente 7": "Lucca (NCD)",
            "Paziente 8": "Tione (NCD)",
            "Paziente 9": "Gemelli - Roma (NCD)",
            "Paziente 10": "Tione (NCD)",
            "Paziente 11": "Gemelli - Roma (NCD)",
            "Paziente 12": "Gemelli - Roma (NCD)",
            "Paziente 13": "Gemelli - Roma (NCD)",
            "Paziente 14": "Gemelli - Roma (NCD)",
        }

        # Se il centro medico è "No Covid Data", usiamo il mapping dei pazienti
        if medical_center == "No Covid Data":
            patient_reference = self.get_patient()
            medical_center = patient_to_medical_center.get(patient_reference, "Unknown")

        return medical_center

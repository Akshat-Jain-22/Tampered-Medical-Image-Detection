# MetaData Issue

def check_metadata_rules(meta):
    issues = []

    if meta["patient_id_len"] == 0:
        issues.append("Missing Patient ID")

    if meta["study_date_valid"] == 0:
        issues.append("Missing scan date")

    if meta["modality_CT"] == 0 and meta["modality_MR"] == 0 :
        issues.append("Unknown modality")

    if meta["slice_thickness"] <= 0:
        issues.append("Invalid slice thickness")

    if meta["pixel_spacing_x"] <= 0 or meta["pixel_spacing_y"] <= 0:
        issues.append("Invalid pixel spacing")
    
    if meta["manufacturer_len"] == 0:
        issues.append("Missing manufacturer info")

    if meta["uid_entropy"] < 2.0:
        issues.append("Low SOP Instance UID entropy")

    return issues

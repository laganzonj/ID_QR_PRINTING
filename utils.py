# utils.py
def validate_config(config):
    """Validate the configuration dictionary"""
    if not isinstance(config, dict):
        return False
    return all(k in config for k in ['active_dataset', 'active_template'])

def validate_dataset(df):
    """Validate the dataset DataFrame"""
    required_columns = {'ID', 'Name', 'Position', 'Company'}
    return required_columns.issubset(df.columns)
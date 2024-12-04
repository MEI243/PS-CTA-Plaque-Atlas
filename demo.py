from build_plaque_atlas import Averaging_Ctl, Plaque_mapping, Visualization

if __name__ == '__main__':
    # Average shape calculation
    Averaging_Ctl().execute()

    # Plaque characteristic mapping
    Plaque_mapping().execute()

    # Atlases visualization
    Visualization().execute()
    

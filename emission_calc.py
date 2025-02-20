import pandas as pd

# Load the CSV file into a DataFrame
ef_df = pd.read_csv('Backend/Emission_Factor_Line_Source.csv')


# Helper function to format emissions (round and keep 4 decimal places)
def format_emission(value):
    return round(value, 4)


# Helper function to get emission factors for a specific vehicle type from the DataFrame
def get_ef_values(vehicle_type):
    
    try: 
        row = ef_df[ef_df['Efactor'] == vehicle_type].iloc[0]
    
        return {
            'BSVI_PM': row['BSVI_PM'], 'BSIV_PM': row['BSIV_PM'], 'BSIII_PM': row['BSIII_PM'],
            'BSVI_NOx': row['BSVI_NOx'], 'BSIV_NOx': row['BSIV_NOx'], 'BSIII_NOx': row['BSIII_NOx'],
            'BSVI_HC': row['BSVI_HC'], 'BSIV_HC': row['BSIV_HC'], 'BSIII_HC': row['BSIII_HC'],
            'BSVI_CO': row['BSVI_CO'], 'BSIV_CO': row['BSIV_CO'], 'BSIII_CO': row['BSIII_CO']
        }      
    except IndexError:
        print(f"Error: {vehicle_type} not found in emission factors.")
        return None  # or handle this case as needed
    
    

# Two-wheelers emission calculation (assumed vehicle type 'Motor Cycle_Petrol')
def calculate_2w_emission(Two_W, RdLength):
    EF_2W = get_ef_values('Motor_Cycle_Petrol')
    
    EI_2W_PM = ((EF_2W['BSIII_PM'] * Two_W * 0.85 * RdLength) + 
                (EF_2W['BSIV_PM'] * Two_W * 0.10 * RdLength) + 
                (EF_2W['BSVI_PM'] * Two_W * 0.05 * RdLength)) * 0.001
    
    EI_2W_NOx = ((EF_2W['BSIII_NOx'] * Two_W * 0.85 * RdLength) + 
                 (EF_2W['BSIV_NOx'] * Two_W * 0.10 * RdLength) + 
                 (EF_2W['BSVI_NOx'] * Two_W * 0.05 * RdLength)) * 0.001
    
    EI_2W_HC = ((EF_2W['BSIII_HC'] * Two_W * 0.85 * RdLength) + 
                (EF_2W['BSIV_HC'] * Two_W * 0.10 * RdLength) + 
                (EF_2W['BSVI_HC'] * Two_W * 0.05 * RdLength)) * 0.001
    
    EI_2W_CO = ((EF_2W['BSIII_CO'] * Two_W * 0.85 * RdLength) + 
                (EF_2W['BSIV_CO'] * Two_W * 0.10 * RdLength) + 
                (EF_2W['BSVI_CO'] * Two_W * 0.05 * RdLength)) * 0.001
    
    return {
        'PM': format_emission(EI_2W_PM),
        'NOx': format_emission(EI_2W_NOx),
        'HC': format_emission(EI_2W_HC),
        'CO': format_emission(EI_2W_CO)
    }
    


# Three-wheelers emission calculation
def calculate_3w_emission(Three_W, RdLength):
    # Get emission factors for each type of three-wheeler
    EF_3W_P = get_ef_values('Three_Wheeler_Petrol')  # Petrol three-wheeler
    EF_3W_D = get_ef_values('Three_Wheeler_Diesel')  # Diesel three-wheeler
    EF_3W_CNG = get_ef_values('Three_Wheeler_CNG')   # CNG three-wheeler

    # Distribution of Three-Wheelers
    Three_W_P = Three_W * 0.40
    Three_W_D = Three_W * 0.20
    Three_W_CNG = Three_W * 0.40

    # PM Emission Calculation
    EI_3WP_PM = ((EF_3W_P['BSIII_PM'] * Three_W_P * 0.85 * RdLength) + 
                  (EF_3W_P['BSIV_PM'] * Three_W_P * 0.10 * RdLength) + 
                  (EF_3W_P['BSVI_PM'] * Three_W_P * 0.05 * RdLength)) * 0.001

    EI_3WD_PM = ((EF_3W_D['BSIII_PM'] * Three_W_D * 0.85 * RdLength) + 
                  (EF_3W_D['BSIV_PM'] * Three_W_D * 0.10 * RdLength) + 
                  (EF_3W_D['BSVI_PM'] * Three_W_D * 0.05 * RdLength)) * 0.001

    EI_3WCNG_PM = ((EF_3W_CNG['BSIII_PM'] * Three_W_CNG * 0.85 * RdLength) + 
                    (EF_3W_CNG['BSIV_PM'] * Three_W_CNG * 0.10 * RdLength) + 
                    (EF_3W_CNG['BSVI_PM'] * Three_W_CNG * 0.05 * RdLength)) * 0.001

    EI_3W_PM = EI_3WP_PM + EI_3WD_PM + EI_3WCNG_PM

    # NOx Emission Calculation
    EI_3WP_NOx = ((EF_3W_P['BSIII_NOx'] * Three_W_P * 0.85 * RdLength) + 
                   (EF_3W_P['BSIV_NOx'] * Three_W_P * 0.10 * RdLength) + 
                   (EF_3W_P['BSVI_NOx'] * Three_W_P * 0.05 * RdLength)) * 0.001

    EI_3WD_NOx = ((EF_3W_D['BSIII_NOx'] * Three_W_D * 0.85 * RdLength) + 
                   (EF_3W_D['BSIV_NOx'] * Three_W_D * 0.10 * RdLength) + 
                   (EF_3W_D['BSVI_NOx'] * Three_W_D * 0.05 * RdLength)) * 0.001

    EI_3WCNG_NOx = ((EF_3W_CNG['BSIII_NOx'] * Three_W_CNG * 0.85 * RdLength) + 
                     (EF_3W_CNG['BSIV_NOx'] * Three_W_CNG * 0.10 * RdLength) + 
                     (EF_3W_CNG['BSVI_NOx'] * Three_W_CNG * 0.05 * RdLength)) * 0.001

    EI_3W_NOx = EI_3WP_NOx + EI_3WD_NOx + EI_3WCNG_NOx

    # HC Emission Calculation
    EI_3WP_HC = ((EF_3W_P['BSIII_HC'] * Three_W_P * 0.85 * RdLength) + 
                  (EF_3W_P['BSIV_HC'] * Three_W_P * 0.10 * RdLength) + 
                  (EF_3W_P['BSVI_HC'] * Three_W_P * 0.05 * RdLength)) * 0.001

    EI_3WD_HC = ((EF_3W_D['BSIII_HC'] * Three_W_D * 0.85 * RdLength) + 
                  (EF_3W_D['BSIV_HC'] * Three_W_D * 0.10 * RdLength) + 
                  (EF_3W_D['BSVI_HC'] * Three_W_D * 0.05 * RdLength)) * 0.001

    EI_3WCNG_HC = ((EF_3W_CNG['BSIII_HC'] * Three_W_CNG * 0.85 * RdLength) + 
                    (EF_3W_CNG['BSIV_HC'] * Three_W_CNG * 0.10 * RdLength) + 
                    (EF_3W_CNG['BSVI_HC'] * Three_W_CNG * 0.05 * RdLength)) * 0.001

    EI_3W_HC = EI_3WP_HC + EI_3WD_HC + EI_3WCNG_HC

    # CO Emission Calculation
    EI_3WP_CO = ((EF_3W_P['BSIII_CO'] * Three_W_P * 0.85 * RdLength) + 
                  (EF_3W_P['BSIV_CO'] * Three_W_P * 0.10 * RdLength) + 
                  (EF_3W_P['BSVI_CO'] * Three_W_P * 0.05 * RdLength)) * 0.001

    EI_3WD_CO = ((EF_3W_D['BSIII_CO'] * Three_W_D * 0.85 * RdLength) + 
                  (EF_3W_D['BSIV_CO'] * Three_W_D * 0.10 * RdLength) + 
                  (EF_3W_D['BSVI_CO'] * Three_W_D * 0.05 * RdLength)) * 0.001

    EI_3WCNG_CO = ((EF_3W_CNG['BSIII_CO'] * Three_W_CNG * 0.85 * RdLength) + 
                    (EF_3W_CNG['BSIV_CO'] * Three_W_CNG * 0.10 * RdLength) + 
                    (EF_3W_CNG['BSVI_CO'] * Three_W_CNG * 0.05 * RdLength)) * 0.001

    EI_3W_CO = EI_3WP_CO + EI_3WD_CO + EI_3WCNG_CO

    return {
        'PM': format_emission(EI_3W_PM),
        'NOx': format_emission(EI_3W_NOx),
        'HC': format_emission(EI_3W_HC),
        'CO': format_emission(EI_3W_CO)
    }


def calculate_4w_emission(Four_W, RdLength):
    # Get emission factors for each type of four-wheeler
    EF_4W_P = get_ef_values('Car_Petrol')  # Petrol four-wheeler
    EF_4W_D = get_ef_values('Car_Diesel')  # Diesel four-wheeler
    EF_4W_CNG = get_ef_values('Car_CNG')    # CNG four-wheeler

    # Distribution of Four-Wheelers
    Four_W_P = Four_W * 0.34
    Four_W_D = Four_W * 0.38
    Four_W_CNG = Four_W * 0.28

    # PM Emission Calculation
    EI_4WP_PM = ((EF_4W_P['BSIII_PM'] * Four_W_P * 0.85 * RdLength) +
                  (EF_4W_P['BSIV_PM'] * Four_W_P * 0.10 * RdLength) +
                  (EF_4W_P['BSVI_PM'] * Four_W_P * 0.05 * RdLength)) * 0.001

    EI_4WD_PM = ((EF_4W_D['BSIII_PM'] * Four_W_D * 0.85 * RdLength) +
                  (EF_4W_D['BSIV_PM'] * Four_W_D * 0.10 * RdLength) +
                  (EF_4W_D['BSVI_PM'] * Four_W_D * 0.05 * RdLength)) * 0.001

    EI_4WCNG_PM = ((EF_4W_CNG['BSIII_PM'] * Four_W_CNG * 0.85 * RdLength) +
                    (EF_4W_CNG['BSIV_PM'] * Four_W_CNG * 0.10 * RdLength) +
                    (EF_4W_CNG['BSVI_PM'] * Four_W_CNG * 0.05 * RdLength)) * 0.001

    EI_4W_PM = EI_4WP_PM + EI_4WD_PM + EI_4WCNG_PM

    # NOx Emission Calculation
    EI_4WP_NOx = ((EF_4W_P['BSIII_NOx'] * Four_W_P * 0.85 * RdLength) +
                   (EF_4W_P['BSIV_NOx'] * Four_W_P * 0.10 * RdLength) +
                   (EF_4W_P['BSVI_NOx'] * Four_W_P * 0.05 * RdLength)) * 0.001

    EI_4WD_NOx = ((EF_4W_D['BSIII_NOx'] * Four_W_D * 0.85 * RdLength) +
                   (EF_4W_D['BSIV_NOx'] * Four_W_D * 0.10 * RdLength) +
                   (EF_4W_D['BSVI_NOx'] * Four_W_D * 0.05 * RdLength)) * 0.001

    EI_4WCNG_NOx = ((EF_4W_CNG['BSIII_NOx'] * Four_W_CNG * 0.85 * RdLength) +
                     (EF_4W_CNG['BSIV_NOx'] * Four_W_CNG * 0.10 * RdLength) +
                     (EF_4W_CNG['BSVI_NOx'] * Four_W_CNG * 0.05 * RdLength)) * 0.001

    EI_4W_NOx = EI_4WP_NOx + EI_4WD_NOx + EI_4WCNG_NOx

    # HC Emission Calculation
    EI_4WP_HC = ((EF_4W_P['BSIII_HC'] * Four_W_P * 0.85 * RdLength) +
                  (EF_4W_P['BSIV_HC'] * Four_W_P * 0.10 * RdLength) +
                  (EF_4W_P['BSVI_HC'] * Four_W_P * 0.05 * RdLength)) * 0.001

    EI_4WD_HC = ((EF_4W_D['BSIII_HC'] * Four_W_D * 0.85 * RdLength) +
                  (EF_4W_D['BSIV_HC'] * Four_W_D * 0.10 * RdLength) +
                  (EF_4W_D['BSVI_HC'] * Four_W_D * 0.05 * RdLength)) * 0.001

    EI_4WCNG_HC = ((EF_4W_CNG['BSIII_HC'] * Four_W_CNG * 0.85 * RdLength) +
                    (EF_4W_CNG['BSIV_HC'] * Four_W_CNG * 0.10 * RdLength) +
                    (EF_4W_CNG['BSVI_HC'] * Four_W_CNG * 0.05 * RdLength)) * 0.001

    EI_4W_HC = EI_4WP_HC + EI_4WD_HC + EI_4WCNG_HC

    # CO Emission Calculation
    EI_4WP_CO = ((EF_4W_P['BSIII_CO'] * Four_W_P * 0.85 * RdLength) +
                  (EF_4W_P['BSIV_CO'] * Four_W_P * 0.10 * RdLength) +
                  (EF_4W_P['BSVI_CO'] * Four_W_P * 0.05 * RdLength)) * 0.001

    EI_4WD_CO = ((EF_4W_D['BSIII_CO'] * Four_W_D * 0.85 * RdLength) +
                  (EF_4W_D['BSIV_CO'] * Four_W_D * 0.10 * RdLength) +
                  (EF_4W_D['BSVI_CO'] * Four_W_D * 0.05 * RdLength)) * 0.001

    EI_4WCNG_CO = ((EF_4W_CNG['BSIII_CO'] * Four_W_CNG * 0.85 * RdLength) +
                    (EF_4W_CNG['BSIV_CO'] * Four_W_CNG * 0.10 * RdLength) +
                    (EF_4W_CNG['BSVI_CO'] * Four_W_CNG * 0.05 * RdLength)) * 0.001

    EI_4W_CO = EI_4WP_CO + EI_4WD_CO + EI_4WCNG_CO

    return {
        'PM': format_emission(EI_4W_PM),
        'NOx': format_emission(EI_4W_NOx),
        'HC': format_emission(EI_4W_HC),
        'CO': format_emission(EI_4W_CO)
    }

def calculate_lcv_emission(LDV, RdLength):
    # Get emission factors for LCVs (Light Commercial Vehicles)
    EF_LDV = get_ef_values('Light_Commercial_Vehicles')

    # PM Emission Calculation
    EI_LDV_PM = ((EF_LDV['BSIII_PM'] * LDV * 0.85 * RdLength) +
                 (EF_LDV['BSIV_PM'] * LDV * 0.10 * RdLength) +
                 (EF_LDV['BSVI_PM'] * LDV * 0.05 * RdLength)) * 0.001

    # NOx Emission Calculation
    EI_LDV_NOx = ((EF_LDV['BSIII_NOx'] * LDV * 0.85 * RdLength) +
                  (EF_LDV['BSIV_NOx'] * LDV * 0.10 * RdLength) +
                  (EF_LDV['BSVI_NOx'] * LDV * 0.05 * RdLength)) * 0.001

    # HC Emission Calculation
    EI_LDV_HC = ((EF_LDV['BSIII_HC'] * LDV * 0.85 * RdLength) +
                 (EF_LDV['BSIV_HC'] * LDV * 0.10 * RdLength) +
                 (EF_LDV['BSVI_HC'] * LDV * 0.05 * RdLength)) * 0.001

    # CO Emission Calculation
    EI_LDV_CO = ((EF_LDV['BSIII_CO'] * LDV * 0.85 * RdLength) +
                 (EF_LDV['BSIV_CO'] * LDV * 0.10 * RdLength) +
                 (EF_LDV['BSVI_CO'] * LDV * 0.05 * RdLength)) * 0.001

    return {
        'PM': format_emission(EI_LDV_PM),
        'NOx': format_emission(EI_LDV_NOx),
        'HC': format_emission(EI_LDV_HC),
        'CO': format_emission(EI_LDV_CO)
    }
    

def calculate_hdv_emission(HDV, RdLength):
    # Get emission factors for HDVs (Heavy-Duty Vehicles)
    EF_HDV = get_ef_values('Heavy_Commercial_Vehicles')

    # PM Emission Calculation
    EI_HDV_PM = ((EF_HDV['BSIII_PM'] * HDV * 0.85 * RdLength) +
                 (EF_HDV['BSIV_PM'] * HDV * 0.10 * RdLength) +
                 (EF_HDV['BSVI_PM'] * HDV * 0.05 * RdLength)) * 0.001

    # NOx Emission Calculation
    EI_HDV_NOx = ((EF_HDV['BSIII_NOx'] * HDV * 0.85 * RdLength) +
                  (EF_HDV['BSIV_NOx'] * HDV * 0.10 * RdLength) +
                  (EF_HDV['BSVI_NOx'] * HDV * 0.05 * RdLength)) * 0.001

    # HC Emission Calculation
    EI_HDV_HC = ((EF_HDV['BSIII_HC'] * HDV * 0.85 * RdLength) +
                 (EF_HDV['BSIV_HC'] * HDV * 0.10 * RdLength) +
                 (EF_HDV['BSVI_HC'] * HDV * 0.05 * RdLength)) * 0.001

    # CO Emission Calculation
    EI_HDV_CO = ((EF_HDV['BSIII_CO'] * HDV * 0.85 * RdLength) +
                 (EF_HDV['BSIV_CO'] * HDV * 0.10 * RdLength) +
                 (EF_HDV['BSVI_CO'] * HDV * 0.05 * RdLength)) * 0.001

    return {
        'PM': format_emission(EI_HDV_PM),
        'NOx': format_emission(EI_HDV_NOx),
        'HC': format_emission(EI_HDV_HC),
        'CO': format_emission(EI_HDV_CO)
    }


# Summing up emissions for all vehicle types
def calculate_total_emission(Two_W, Three_W, Four_W, LDV, HDV, RdLength):
    emissions_2W = calculate_2w_emission(Two_W, RdLength)
    emissions_3W = calculate_3w_emission(Three_W, RdLength)
    emissions_4W = calculate_4w_emission(Four_W, RdLength)
    emissions_LDV = calculate_lcv_emission(LDV, RdLength)
    emissions_HDV = calculate_hdv_emission(HDV, RdLength)
    
    print(f'2W : {emissions_2W}')
    print(f'3W : {emissions_3W}')
    print(f'4W : {emissions_4W}')
    print(f'LDV : {emissions_LDV}')
    print(f'HDV : {emissions_HDV}')
    
    total_PM = emissions_2W['PM'] + emissions_3W['PM'] + emissions_4W['PM'] + emissions_LDV['PM'] + emissions_HDV['PM']
    total_NOx = emissions_2W['NOx'] + emissions_3W['NOx'] + emissions_4W['NOx'] + emissions_LDV['NOx'] + emissions_HDV['NOx']
    total_HC = emissions_2W['HC'] + emissions_3W['HC'] + emissions_4W['HC'] + emissions_LDV['HC'] + emissions_HDV['HC']
    total_CO = emissions_2W['CO'] + emissions_3W['CO'] + emissions_4W['CO'] + emissions_LDV['CO'] + emissions_HDV['CO']
    
    return {
        'Total_PM': format_emission(total_PM),
        'Total_NOx': format_emission(total_NOx),
        'Total_HC': format_emission(total_HC),
        'Total_CO': format_emission(total_CO)
    }

# Example of how to call the function:
emissions = calculate_total_emission(2000, 1453, 980, 300, 73, RdLength=5.0)  
print(f'total : {emissions}')
from skylines.dataset import Dominance
from skylines.dataset.real.real import RealDataset


class AutoMPG_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='autompg',
                         control=['mpg', 'cylinders', 'horsepower', 'acceleration'],
                         preference=['weight', 'displacement'],
                         effect={
                             'mpg': {'weight': -1, 'displacement': -1, 'horsepower': -1},
                             'horsepower': {'displacement': 1},
                             'acceleration': {'horsepower': -1, 'weight': 1},
                             'displacement': {'cylinders': 1},
                             'weight': {'cylinders': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'displacement': Dominance.MAX,
                             'weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class AutoMPG_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='autompg',
                         control=['mpg', 'cylinders', 'horsepower', 'weight'],
                         preference=['displacement', 'acceleration'],
                         effect={
                             'mpg': {'weight': -1, 'displacement': -1, 'horsepower': -1},
                             'horsepower': {'displacement': 1},
                             'acceleration': {'horsepower': -1, 'weight': 1},
                             'displacement': {'cylinders': 1},
                             'weight': {'cylinders': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'acceleration': Dominance.MAX,
                             'displacement': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class AutoMPG_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='autompg',
                         control=['mpg', 'cylinders', 'horsepower'],
                         preference=['weight', 'displacement', 'acceleration'],
                         effect={
                             'mpg': {'weight': -1, 'displacement': -1, 'horsepower': -1},
                             'horsepower': {'displacement': 1},
                             'acceleration': {'horsepower': -1, 'weight': 1},
                             'displacement': {'cylinders': 1},
                             'weight': {'cylinders': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'acceleration': Dominance.MAX,
                             'displacement': Dominance.MAX,
                             'weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class AutoMPG_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='autompg',
                         control=['mpg', 'horsepower'],
                         preference=['weight', 'displacement', 'acceleration', 'cylinders'],
                         effect={
                             'mpg': {'weight': -1, 'displacement': -1, 'horsepower': -1},
                             'horsepower': {'displacement': 1},
                             'acceleration': {'horsepower': -1, 'weight': 1},
                             'displacement': {'cylinders': 1},
                             'weight': {'cylinders': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'acceleration': Dominance.MAX,
                             'cylinders': Dominance.MAX,
                             'displacement': Dominance.MAX,
                             'weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class AutoMPG_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='autompg',
                         control=['mpg'],
                         preference=['weight', 'displacement', 'acceleration', 'cylinders', 'horsepower'],
                         effect={
                             'mpg': {'weight': -1, 'displacement': -1, 'horsepower': -1},
                             'horsepower': {'displacement': 1},
                             'acceleration': {'horsepower': -1, 'weight': 1},
                             'displacement': {'cylinders': 1},
                             'weight': {'cylinders': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'acceleration': Dominance.MAX,
                             'cylinders': Dominance.MAX,
                             'displacement': Dominance.MAX,
                             'horsepower': Dominance.MAX,
                             'weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)








class Hong_Kong_Weather_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='hong_kong_weather',
                         control=['mean_temp', 'max_temp', 'min_grass_temp', 'min_temp', 'mean_dew_point_temp', 'mean_wet_bulb_temp', 'rainfall', 'pressure'],
                         preference=['rel_humidity', 'cloud'],
                         effect={
                             'min_temp': {'cloud': 1},
                             'max_temp': {'cloud': -1},
                             'mean_temp': {'min_temp': 1, 'max_temp': 1},
                             'pressure': {'mean_temp': -1},
                             'rel_humidity': {'mean_temp': -1, 'mean_dew_point_temp': 1},
                             'cloud': {'rel_humidity': 1},
                             'rainfall': {'cloud': 1, 'rel_humidity': 1, 'pressure': -1},
                             'mean_wet_bulb_temp': {'mean_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1},
                             'min_grass_temp': {'min_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1, 'cloud': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'cloud': Dominance.MAX,
                             'rel_humidity': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Hong_Kong_Weather_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='hong_kong_weather',
                         control=['mean_temp', 'max_temp', 'min_grass_temp', 'min_temp', 'mean_dew_point_temp', 'mean_wet_bulb_temp', 'rainfall', 'cloud'],
                         preference=['rel_humidity', 'pressure'],
                         effect={
                             'min_temp': {'cloud': 1},
                             'max_temp': {'cloud': -1},
                             'mean_temp': {'min_temp': 1, 'max_temp': 1},
                             'pressure': {'mean_temp': -1},
                             'rel_humidity': {'mean_temp': -1, 'mean_dew_point_temp': 1},
                             'cloud': {'rel_humidity': 1},
                             'rainfall': {'cloud': 1, 'rel_humidity': 1, 'pressure': -1},
                             'mean_wet_bulb_temp': {'mean_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1},
                             'min_grass_temp': {'min_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1, 'cloud': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'pressure': Dominance.MAX,
                             'rel_humidity': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Hong_Kong_Weather_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='hong_kong_weather',
                         control=['mean_temp', 'max_temp', 'min_grass_temp', 'min_temp', 'mean_dew_point_temp', 'mean_wet_bulb_temp', 'rainfall'],
                         preference=['rel_humidity', 'pressure', 'cloud'],
                         effect={
                             'min_temp': {'cloud': 1},
                             'max_temp': {'cloud': -1},
                             'mean_temp': {'min_temp': 1, 'max_temp': 1},
                             'pressure': {'mean_temp': -1},
                             'rel_humidity': {'mean_temp': -1, 'mean_dew_point_temp': 1},
                             'cloud': {'rel_humidity': 1},
                             'rainfall': {'cloud': 1, 'rel_humidity': 1, 'pressure': -1},
                             'mean_wet_bulb_temp': {'mean_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1},
                             'min_grass_temp': {'min_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1, 'cloud': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'cloud': Dominance.MAX,
                             'pressure': Dominance.MAX,
                             'rel_humidity': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Hong_Kong_Weather_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='hong_kong_weather',
                         control=['mean_temp', 'max_temp', 'min_grass_temp', 'min_temp', 'mean_dew_point_temp', 'cloud'],
                         preference=['rel_humidity', 'mean_wet_bulb_temp', 'rainfall', 'pressure'],
                         effect={
                             'min_temp': {'cloud': 1},
                             'max_temp': {'cloud': -1},
                             'mean_temp': {'min_temp': 1, 'max_temp': 1},
                             'pressure': {'mean_temp': -1},
                             'rel_humidity': {'mean_temp': -1, 'mean_dew_point_temp': 1},
                             'cloud': {'rel_humidity': 1},
                             'rainfall': {'cloud': 1, 'rel_humidity': 1, 'pressure': -1},
                             'mean_wet_bulb_temp': {'mean_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1},
                             'min_grass_temp': {'min_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1, 'cloud': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'mean_wet_bulb_temp': Dominance.MAX,
                             'pressure': Dominance.MAX,
                             'rainfall': Dominance.MAX,
                             'rel_humidity': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class Hong_Kong_Weather_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='hong_kong_weather',
                         control=['mean_temp', 'max_temp', 'min_grass_temp', 'min_temp', 'mean_dew_point_temp'],
                         preference=['rel_humidity', 'mean_wet_bulb_temp', 'rainfall', 'pressure', 'cloud'],
                         effect={
                             'min_temp': {'cloud': 1},
                             'max_temp': {'cloud': -1},
                             'mean_temp': {'min_temp': 1, 'max_temp': 1},
                             'pressure': {'mean_temp': -1},
                             'rel_humidity': {'mean_temp': -1, 'mean_dew_point_temp': 1},
                             'cloud': {'rel_humidity': 1},
                             'rainfall': {'cloud': 1, 'rel_humidity': 1, 'pressure': -1},
                             'mean_wet_bulb_temp': {'mean_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1},
                             'min_grass_temp': {'min_temp': 1, 'rel_humidity': 1, 'mean_dew_point_temp': 1, 'cloud': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'cloud': Dominance.MAX,
                             'mean_wet_bulb_temp': Dominance.MAX,
                             'pressure': Dominance.MAX,
                             'rainfall': Dominance.MAX,
                             'rel_humidity': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)









class DryBean_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='drybean',
                         control=['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'EquivDiameter', 'Compactness', 'Roundness', 'AspectRatio'],
                         preference=['Perimeter', 'Area'],
                         effect={
                             'Area': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'Perimeter': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'AspectRatio': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Eccentricity': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Roundness': {'Area': 1, 'Perimeter': -1},
                             'Compactness': {'Perimeter': -1, 'EquivDiameter': 1},
                             'EquivDiameter': {'AspectRatio': -1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'Area': Dominance.MAX,
                             'Perimeter': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class DryBean_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='drybean',
                         control=['MajorAxisLength', 'MinorAxisLength', 'Area', 'Eccentricity', 'EquivDiameter', 'AspectRatio', 'Compactness'],
                         preference=['Perimeter', 'Roundness'],
                         effect={
                             'Area': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'Perimeter': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'AspectRatio': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Eccentricity': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Roundness': {'Area': 1, 'Perimeter': -1},
                             'Compactness': {'Perimeter': -1, 'EquivDiameter': 1},
                             'EquivDiameter': {'AspectRatio': -1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'Perimeter': Dominance.MAX,
                             'Roundness': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class DryBean_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='drybean',
                         control=['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'EquivDiameter', 'AspectRatio', 'Compactness'],
                         preference=['Area', 'Perimeter', 'Roundness'],
                         effect={
                             'Area': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'Perimeter': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'AspectRatio': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Eccentricity': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Roundness': {'Area': 1, 'Perimeter': -1},
                             'Compactness': {'Perimeter': -1, 'EquivDiameter': 1},
                             'EquivDiameter': {'AspectRatio': -1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'Area': Dominance.MAX,
                             'Perimeter': Dominance.MAX,
                             'Roundness': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class DryBean_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='drybean',
                         control=['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'EquivDiameter', 'AspectRatio'],
                         preference=['Area', 'Perimeter', 'Roundness', 'Compactness'],
                         effect={
                             'Area': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'Perimeter': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'AspectRatio': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Eccentricity': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Roundness': {'Area': 1, 'Perimeter': -1},
                             'Compactness': {'Perimeter': -1, 'EquivDiameter': 1},
                             'EquivDiameter': {'AspectRatio': -1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'Area': Dominance.MAX,
                             'Perimeter': Dominance.MAX,
                             'Roundness': Dominance.MAX,
                             'Compactness': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class DryBean_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='drybean',
                         control=['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'EquivDiameter'],
                         preference=['Area', 'Perimeter', 'Roundness', 'Compactness', 'AspectRatio'],
                         effect={
                             'Area': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'Perimeter': {'MajorAxisLength': 1, 'MinorAxisLength': 1},
                             'AspectRatio': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Eccentricity': {'MajorAxisLength': 1, 'MinorAxisLength': -1},
                             'Roundness': {'Area': 1, 'Perimeter': -1},
                             'Compactness': {'Perimeter': -1, 'EquivDiameter': 1},
                             'EquivDiameter': {'AspectRatio': -1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'Area': Dominance.MAX,
                             'Perimeter': Dominance.MAX,
                             'Roundness': Dominance.MAX,
                             'Compactness': Dominance.MAX,
                             'AspectRatio': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)







class IndividualHouseholdPowerConsumption_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 2_049_280,
                 seed: int = 42):
        super().__init__(file_name='individual_household_power_consumption',
                         control=['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Voltage'],
                         preference=['Sub_metering_3', 'Global_intensity'],
                         effect={
                             'Global_intensity': {'Voltage': 1, 'Global_apparent_power': 1},
                             'Global_apparent_power': {'Global_reactive_power': 1, 'Global_active_power': 1},
                             'Global_reactive_power': {'Global_active_power': -1, 'Phase Angle': -1},
                             'Global_active_power': {'Sub_metering_1': 1, 'Sub_metering_2': 1, 'Sub_metering_3': 1},
                             'Sub_metering_1': {'Date': 0, 'Time': 0},
                             'Sub_metering_2': {'Date': 0, 'Time': 0},
                             'Sub_metering_3': {'Date': 0, 'Time': 0}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Global_intensity': Dominance.MAX,
                             'Sub_metering_3': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class IndividualHouseholdPowerConsumption_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 2_049_280,
                 seed: int = 42):
        super().__init__(file_name='individual_household_power_consumption',
                         control=['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
                         preference=['Global_intensity', 'Voltage'],
                         effect={
                             'Global_intensity': {'Voltage': 1, 'Global_apparent_power': 1},
                             'Global_apparent_power': {'Global_reactive_power': 1, 'Global_active_power': 1},
                             'Global_reactive_power': {'Global_active_power': -1, 'Phase Angle': -1},
                             'Global_active_power': {'Sub_metering_1': 1, 'Sub_metering_2': 1, 'Sub_metering_3': 1},
                             'Sub_metering_1': {'Date': 0, 'Time': 0},
                             'Sub_metering_2': {'Date': 0, 'Time': 0},
                             'Sub_metering_3': {'Date': 0, 'Time': 0}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Global_intensity': Dominance.MAX,
                             'Voltage': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class IndividualHouseholdPowerConsumption_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 2_049_280,
                 seed: int = 42):
        super().__init__(file_name='individual_household_power_consumption',
                         control=['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2'],
                         preference=['Sub_metering_3', 'Global_intensity', 'Voltage'],
                         effect={
                             'Global_intensity': {'Voltage': 1, 'Global_apparent_power': 1},
                             'Global_apparent_power': {'Global_reactive_power': 1, 'Global_active_power': 1},
                             'Global_reactive_power': {'Global_active_power': -1, 'Phase Angle': -1},
                             'Global_active_power': {'Sub_metering_1': 1, 'Sub_metering_2': 1, 'Sub_metering_3': 1},
                             'Sub_metering_1': {'Date': 0, 'Time': 0},
                             'Sub_metering_2': {'Date': 0, 'Time': 0},
                             'Sub_metering_3': {'Date': 0, 'Time': 0}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Global_intensity': Dominance.MAX,
                             'Sub_metering_3': Dominance.MAX,
                             'Voltage': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class IndividualHouseholdPowerConsumption_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 2_049_280,
                 seed: int = 42):
        super().__init__(file_name='individual_household_power_consumption',
                         control=['Global_active_power', 'Global_reactive_power', 'Sub_metering_1'],
                         preference=['Sub_metering_2', 'Sub_metering_3', 'Global_intensity', 'Voltage'],
                         effect={
                             'Global_intensity': {'Voltage': 1, 'Global_apparent_power': 1},
                             'Global_apparent_power': {'Global_reactive_power': 1, 'Global_active_power': 1},
                             'Global_reactive_power': {'Global_active_power': -1, 'Phase Angle': -1},
                             'Global_active_power': {'Sub_metering_1': 1, 'Sub_metering_2': 1, 'Sub_metering_3': 1},
                             'Sub_metering_1': {'Date': 0, 'Time': 0},
                             'Sub_metering_2': {'Date': 0, 'Time': 0},
                             'Sub_metering_3': {'Date': 0, 'Time': 0}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Global_intensity': Dominance.MAX,
                             'Sub_metering_2': Dominance.MAX,
                             'Sub_metering_3': Dominance.MAX,
                             'Voltage': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class IndividualHouseholdPowerConsumption_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 2_049_280,
                 seed: int = 42):
        super().__init__(file_name='individual_household_power_consumption',
                         control=['Global_active_power', 'Global_reactive_power'],
                         preference=['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Global_intensity', 'Voltage'],
                         effect={
                             'Global_intensity': {'Voltage': 1, 'Global_apparent_power': 1},
                             'Global_apparent_power': {'Global_reactive_power': 1, 'Global_active_power': 1},
                             'Global_reactive_power': {'Global_active_power': -1, 'Phase Angle': -1},
                             'Global_active_power': {'Sub_metering_1': 1, 'Sub_metering_2': 1, 'Sub_metering_3': 1},
                             'Sub_metering_1': {'Date': 0, 'Time': 0},
                             'Sub_metering_2': {'Date': 0, 'Time': 0},
                             'Sub_metering_3': {'Date': 0, 'Time': 0}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Global_intensity': Dominance.MAX,
                             'Sub_metering_1': Dominance.MAX,
                             'Sub_metering_2': Dominance.MAX,
                             'Sub_metering_3': Dominance.MAX,
                             'Voltage': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)











class Abalone_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='abalone',
                         control=['Rings', 'Diameter', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Height', 'Sex'],
                         preference=['Length', 'Whole Weight'],
                         effect={
                             'Rings': {'Age': 1},
                             'Size/Volume': {'Age': 1, 'Sex': 0},
                             'Length': {'Size/Volume': 1},
                             'Diameter': {'Size/Volume': 1},
                             'Height': {'Size/Volume': 1},
                             'Shucked Weight': {'Size/Volume': 1},
                             'Viscera Weight': {'Size/Volume': 1},
                             'Shell Weight': {'Size/Volume': 1},
                             'Whole Weight': {'Viscera Weight': 1, 'Shell Weight': 1, 'Shucked Weight': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Length': Dominance.MAX,
                             'Whole Weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Abalone_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='abalone',
                         control=['Rings', 'Diameter', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Height', 'Whole Weight'],
                         preference=['Length', 'Sex'],
                         effect={
                             'Rings': {'Age': 1},
                             'Size/Volume': {'Age': 1, 'Sex': 0},
                             'Length': {'Size/Volume': 1},
                             'Diameter': {'Size/Volume': 1},
                             'Height': {'Size/Volume': 1},
                             'Shucked Weight': {'Size/Volume': 1},
                             'Viscera Weight': {'Size/Volume': 1},
                             'Shell Weight': {'Size/Volume': 1},
                             'Whole Weight': {'Viscera Weight': 1, 'Shell Weight': 1, 'Shucked Weight': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Length': Dominance.MAX,
                             'Sex': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Abalone_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='abalone',
                         control=['Rings', 'Diameter', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Height'],
                         preference=['Length', 'Whole Weight', 'Sex'],
                         effect={
                             'Rings': {'Age': 1},
                             'Size/Volume': {'Age': 1, 'Sex': 0},
                             'Length': {'Size/Volume': 1},
                             'Diameter': {'Size/Volume': 1},
                             'Height': {'Size/Volume': 1},
                             'Shucked Weight': {'Size/Volume': 1},
                             'Viscera Weight': {'Size/Volume': 1},
                             'Shell Weight': {'Size/Volume': 1},
                             'Whole Weight': {'Viscera Weight': 1, 'Shell Weight': 1, 'Shucked Weight': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Length': Dominance.MAX,
                             'Sex': Dominance.MAX,
                             'Whole Weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Abalone_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='abalone',
                         control=['Rings', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Height'],
                         preference=['Diameter', 'Length', 'Whole Weight', 'Sex'],
                         effect={
                             'Rings': {'Age': 1},
                             'Size/Volume': {'Age': 1, 'Sex': 0},
                             'Length': {'Size/Volume': 1},
                             'Diameter': {'Size/Volume': 1},
                             'Height': {'Size/Volume': 1},
                             'Shucked Weight': {'Size/Volume': 1},
                             'Viscera Weight': {'Size/Volume': 1},
                             'Shell Weight': {'Size/Volume': 1},
                             'Whole Weight': {'Viscera Weight': 1, 'Shell Weight': 1, 'Shucked Weight': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Diameter': Dominance.MAX,
                             'Length': Dominance.MAX,
                             'Sex': Dominance.MAX,
                             'Whole Weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class Abalone_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='abalone',
                         control=['Rings', 'Shucked Weight', 'Viscera Weight', 'Shell Weight'],
                         preference=['Height', 'Diameter', 'Length', 'Whole Weight', 'Sex'],
                         effect={
                             'Rings': {'Age': 1},
                             'Size/Volume': {'Age': 1, 'Sex': 0},
                             'Length': {'Size/Volume': 1},
                             'Diameter': {'Size/Volume': 1},
                             'Height': {'Size/Volume': 1},
                             'Shucked Weight': {'Size/Volume': 1},
                             'Viscera Weight': {'Size/Volume': 1},
                             'Shell Weight': {'Size/Volume': 1},
                             'Whole Weight': {'Viscera Weight': 1, 'Shell Weight': 1, 'Shucked Weight': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Diameter': Dominance.MAX,
                             'Height': Dominance.MAX,
                             'Length': Dominance.MAX,
                             'Sex': Dominance.MAX,
                             'Whole Weight': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)













class ConcreteCompressiveStrength_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='concrete_compressive_strength',
                         control=['cement', 'water', 'coarse aggregate', 'fine aggregate', 'age', 'concrete compressive strength', 'blast furnace slag'],
                         preference=['fly ash', 'superplasticizer'],
                         effect={
                             'concrete compressive strength': {
                                 'cement': 1,
                                 'blast furnace slag': 1,
                                 'fly ash': 1,
                                 'water': -1,
                                 'superplasticizer': 1,
                                 'coarse aggregate': 1,
                                 'fine aggregate': 1,
                                 'age': 1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'fly ash': Dominance.MAX,
                             'superplasticizer': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class ConcreteCompressiveStrength_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='concrete_compressive_strength',
                         control=['cement', 'water', 'coarse aggregate', 'fine aggregate', 'age', 'concrete compressive strength', 'superplasticizer'],
                         preference=['blast furnace slag', 'fly ash'],
                         effect={
                             'concrete compressive strength': {
                                 'cement': 1,
                                 'blast furnace slag': 1,
                                 'fly ash': 1,
                                 'water': -1,
                                 'superplasticizer': 1,
                                 'coarse aggregate': 1,
                                 'fine aggregate': 1,
                                 'age': 1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'blast furnace slag': Dominance.MAX,
                             'fly ash': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class ConcreteCompressiveStrength_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='concrete_compressive_strength',
                         control=['cement', 'water', 'coarse aggregate', 'fine aggregate', 'age', 'concrete compressive strength'],
                         preference=['blast furnace slag', 'fly ash', 'superplasticizer'],
                         effect={
                             'concrete compressive strength': {
                                 'cement': 1,
                                 'blast furnace slag': 1,
                                 'fly ash': 1,
                                 'water': -1,
                                 'superplasticizer': 1,
                                 'coarse aggregate': 1,
                                 'fine aggregate': 1,
                                 'age': 1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'blast furnace slag': Dominance.MAX,
                             'fly ash': Dominance.MAX,
                             'superplasticizer': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class ConcreteCompressiveStrength_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='concrete_compressive_strength',
                         control=['coarse aggregate', 'fine aggregate', 'age', 'concrete compressive strength', 'water'],
                         preference=['blast furnace slag', 'fly ash', 'superplasticizer', 'cement'],
                         effect={
                             'concrete compressive strength': {
                                 'cement': 1,
                                 'blast furnace slag': 1,
                                 'fly ash': 1,
                                 'water': -1,
                                 'superplasticizer': 1,
                                 'coarse aggregate': 1,
                                 'fine aggregate': 1,
                                 'age': 1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'blast furnace slag': Dominance.MAX,
                             'cement': Dominance.MAX,
                             'fly ash': Dominance.MAX,
                             'superplasticizer': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class ConcreteCompressiveStrength_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='concrete_compressive_strength',
                         control=['coarse aggregate', 'fine aggregate', 'concrete compressive strength', 'water'],
                         preference=['blast furnace slag', 'fly ash', 'superplasticizer', 'cement', 'age'],
                         effect={
                             'concrete compressive strength': {
                                 'cement': 1,
                                 'blast furnace slag': 1,
                                 'fly ash': 1,
                                 'water': -1,
                                 'superplasticizer': 1,
                                 'coarse aggregate': 1,
                                 'fine aggregate': 1,
                                 'age': 1},
                         },
                         dominance=dominance,
                         infer_controls={
                             'age': Dominance.MAX,
                             'blast furnace slag': Dominance.MAX,
                             'cement': Dominance.MAX,
                             'fly ash': Dominance.MAX,
                             'superplasticizer': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)









class SeoulBikeDemandDataset_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='seoul_bike_demand',
                         control=['Temperature(C)', 'Wind speed (m/s)',
                                  'Solar Radiation (MJ/m2)', 'Visibility(10m)', 'Holiday',
                                  'Functioning Day', 'Humidity(%)', 'Dew point temperature(C)',
                                  'Rainfall(mm)', 'Snowfall (cm)'],
                         preference=['Hour', 'Rented Bike Count'],
                         effect={
                             'Season': {'Date': 0},
                             'Wind speed (m/s)': {'Season': 0},
                             'Solar Radiation (MJ/m2)': {'Season': 0, 'Hour': 0},
                             'Humidity(%)': {'Temperature(C)': 1, 'Wind speed (m/s)': 0},
                             'Temperature(C)': {'Wind speed (m/s)': 0, 'Solar Radiation (MJ/m2)': 1},
                             'Holiday': {'Date': 0, 'Season': 0},
                             'Visibility(10m)': {'Rainfall(mm)': -1, 'Snowfall (cm)': -1, 'Wind speed (m/s)': -1,
                                                 'Solar Radiation (MJ/m2)': 1},
                             'Dew point temperature(C)': {'Temperature(C)': 1, 'Humidity(%)': 1},
                             'Snowfall (cm)': {'Temperature(C)': -1, 'Humidity(%)': 1, 'Dew point temperature(C)': 1},
                             'Rainfall(mm)': {'Temperature': 1, 'Dew point temperature(C)': 1, 'Humidity(%)': 1},
                             'Functioning Day': {'Date': 0},
                             'Rented Bike Count': {'Hour': 0,
                                                   'Wind speed (m/s)': -1,
                                                   'Visibility(10m)': 1,
                                                   'Humidity(%)': -1,
                                                   'Temperature(C)': 0,
                                                   'Holiday': 1,
                                                   'Rainfall(mm)': -1,
                                                   'Snowfall (cm)': -1,
                                                   'Functioning Day': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Rented Bike Count': Dominance.MAX,
                             'Hour': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class SeoulBikeDemandDataset_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='seoul_bike_demand',
                         control=['Temperature(C)', 'Wind speed (m/s)',
                                  'Solar Radiation (MJ/m2)', 'Visibility(10m)', 'Holiday',
                                  'Functioning Day', 'Dew point temperature(C)',
                                  'Rainfall(mm)', 'Snowfall (cm)', 'Rented Bike Count'],
                         preference=['Hour', 'Humidity(%)'],
                         effect={
                             'Season': {'Date': 0},
                             'Wind speed (m/s)': {'Season': 0},
                             'Solar Radiation (MJ/m2)': {'Season': 0, 'Hour': 0},
                             'Humidity(%)': {'Temperature(C)': 1, 'Wind speed (m/s)': 0},
                             'Temperature(C)': {'Wind speed (m/s)': 0, 'Solar Radiation (MJ/m2)': 1},
                             'Holiday': {'Date': 0, 'Season': 0},
                             'Visibility(10m)': {'Rainfall(mm)': -1, 'Snowfall (cm)': -1, 'Wind speed (m/s)': -1,
                                                 'Solar Radiation (MJ/m2)': 1},
                             'Dew point temperature(C)': {'Temperature(C)': 1, 'Humidity(%)': 1},
                             'Snowfall (cm)': {'Temperature(C)': -1, 'Humidity(%)': 1, 'Dew point temperature(C)': 1},
                             'Rainfall(mm)': {'Temperature': 1, 'Dew point temperature(C)': 1, 'Humidity(%)': 1},
                             'Functioning Day': {'Date': 0},
                             'Rented Bike Count': {'Hour': 0,
                                                   'Wind speed (m/s)': -1,
                                                   'Visibility(10m)': 1,
                                                   'Humidity(%)': -1,
                                                   'Temperature(C)': 0,
                                                   'Holiday': 1,
                                                   'Rainfall(mm)': -1,
                                                   'Snowfall (cm)': -1,
                                                   'Functioning Day': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Hour': Dominance.MAX,
                             'Humidity(%)': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class SeoulBikeDemandDataset_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='seoul_bike_demand',
                         control=['Temperature(C)', 'Wind speed (m/s)',
                                  'Solar Radiation (MJ/m2)', 'Visibility(10m)', 'Holiday',
                                  'Functioning Day', 'Dew point temperature(C)',
                                  'Rainfall(mm)', 'Snowfall (cm)'],
                         preference=['Hour', 'Humidity(%)', 'Rented Bike Count'],
                         effect={
                             'Season': {'Date': 0},
                             'Wind speed (m/s)': {'Season': 0},
                             'Solar Radiation (MJ/m2)': {'Season': 0, 'Hour': 0},
                             'Humidity(%)': {'Temperature(C)': 1, 'Wind speed (m/s)': 0},
                             'Temperature(C)': {'Wind speed (m/s)': 0, 'Solar Radiation (MJ/m2)': 1},
                             'Holiday': {'Date': 0, 'Season': 0},
                             'Visibility(10m)': {'Rainfall(mm)': -1, 'Snowfall (cm)': -1, 'Wind speed (m/s)': -1,
                                                 'Solar Radiation (MJ/m2)': 1},
                             'Dew point temperature(C)': {'Temperature(C)': 1, 'Humidity(%)': 1},
                             'Snowfall (cm)': {'Temperature(C)': -1, 'Humidity(%)': 1, 'Dew point temperature(C)': 1},
                             'Rainfall(mm)': {'Temperature': 1, 'Dew point temperature(C)': 1, 'Humidity(%)': 1},
                             'Functioning Day': {'Date': 0},
                             'Rented Bike Count': {'Hour': 0,
                                                   'Wind speed (m/s)': -1,
                                                   'Visibility(10m)': 1,
                                                   'Humidity(%)': -1,
                                                   'Temperature(C)': 0,
                                                   'Holiday': 1,
                                                   'Rainfall(mm)': -1,
                                                   'Snowfall (cm)': -1,
                                                   'Functioning Day': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Rented Bike Count': Dominance.MAX,
                             'Hour': Dominance.MAX,
                             'Humidity(%)': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class SeoulBikeDemandDataset_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='seoul_bike_demand',
                         control=['Temperature(C)',
                                  'Solar Radiation (MJ/m2)', 'Visibility(10m)', 'Holiday',
                                  'Functioning Day', 'Dew point temperature(C)',
                                  'Rainfall(mm)', 'Snowfall (cm)'],
                         preference=['Hour', 'Humidity(%)', 'Rented Bike Count', 'Wind speed (m/s)'],
                         effect={
                             'Season': {'Date': 0},
                             'Wind speed (m/s)': {'Season': 0},
                             'Solar Radiation (MJ/m2)': {'Season': 0, 'Hour': 0},
                             'Humidity(%)': {'Temperature(C)': 1, 'Wind speed (m/s)': 0},
                             'Temperature(C)': {'Wind speed (m/s)': 0, 'Solar Radiation (MJ/m2)': 1},
                             'Holiday': {'Date': 0, 'Season': 0},
                             'Visibility(10m)': {'Rainfall(mm)': -1, 'Snowfall (cm)': -1, 'Wind speed (m/s)': -1,
                                                 'Solar Radiation (MJ/m2)': 1},
                             'Dew point temperature(C)': {'Temperature(C)': 1, 'Humidity(%)': 1},
                             'Snowfall (cm)': {'Temperature(C)': -1, 'Humidity(%)': 1, 'Dew point temperature(C)': 1},
                             'Rainfall(mm)': {'Temperature': 1, 'Dew point temperature(C)': 1, 'Humidity(%)': 1},
                             'Functioning Day': {'Date': 0},
                             'Rented Bike Count': {'Hour': 0,
                                                   'Wind speed (m/s)': -1,
                                                   'Visibility(10m)': 1,
                                                   'Humidity(%)': -1,
                                                   'Temperature(C)': 0,
                                                   'Holiday': 1,
                                                   'Rainfall(mm)': -1,
                                                   'Snowfall (cm)': -1,
                                                   'Functioning Day': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Rented Bike Count': Dominance.MAX,
                             'Hour': Dominance.MAX,
                             'Humidity(%)': Dominance.MAX,
                             'Wind speed (m/s)': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
        

class SeoulBikeDemandDataset_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = None,
                 seed: int = 42):
        super().__init__(file_name='seoul_bike_demand',
                         control=['Solar Radiation (MJ/m2)', 'Visibility(10m)', 'Holiday',
                                  'Functioning Day', 'Dew point temperature(C)',
                                  'Rainfall(mm)', 'Snowfall (cm)'],
                         preference=['Hour', 'Humidity(%)', 'Rented Bike Count', 'Wind speed (m/s)', 'Temperature(C)'],
                         effect={
                             'Season': {'Date': 0},
                             'Wind speed (m/s)': {'Season': 0},
                             'Solar Radiation (MJ/m2)': {'Season': 0, 'Hour': 0},
                             'Humidity(%)': {'Temperature(C)': 1, 'Wind speed (m/s)': 0},
                             'Temperature(C)': {'Wind speed (m/s)': 0, 'Solar Radiation (MJ/m2)': 1},
                             'Holiday': {'Date': 0, 'Season': 0},
                             'Visibility(10m)': {'Rainfall(mm)': -1, 'Snowfall (cm)': -1, 'Wind speed (m/s)': -1,
                                                 'Solar Radiation (MJ/m2)': 1},
                             'Dew point temperature(C)': {'Temperature(C)': 1, 'Humidity(%)': 1},
                             'Snowfall (cm)': {'Temperature(C)': -1, 'Humidity(%)': 1, 'Dew point temperature(C)': 1},
                             'Rainfall(mm)': {'Temperature': 1, 'Dew point temperature(C)': 1, 'Humidity(%)': 1},
                             'Functioning Day': {'Date': 0},
                             'Rented Bike Count': {'Hour': 0,
                                                   'Wind speed (m/s)': -1,
                                                   'Visibility(10m)': 1,
                                                   'Humidity(%)': -1,
                                                   'Temperature(C)': 0,
                                                   'Holiday': 1,
                                                   'Rainfall(mm)': -1,
                                                   'Snowfall (cm)': -1,
                                                   'Functioning Day': 1}
                         },
                         dominance=dominance,
                         infer_controls={
                             'Rented Bike Count': Dominance.MAX,
                             'Hour': Dominance.MAX,
                             'Humidity(%)': Dominance.MAX,
                             'Temperature(C)': Dominance.MAX,
                             'Wind speed (m/s)': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)
















class WineQualityWhite_2var_pos(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10_000,
                 seed: int = 42):
        super().__init__(file_name='wine_quality_white',
                         control=['volatile acidity','residual sugar','fixed acidity','total sulfur dioxide','density','pH','sulphates','chlorides','free sulfur dioxide','citric acid'],
                         preference=['alcohol','quality'],
                         effect={'quality': {'chlorides': -1, 'total sulfur dioxide': 1, 'pH': -1, 'alcohol': 0},
                                 'total sulfur dioxide': {'free sulfur dioxide': 1, 'reacted sulfur dioxide': 1},
                                 'pH': {'volatile acidity': -1, 'fixed acidity': -1},
                                 'fixed acidity': {'citric acid': +1},
                                 'alcohol': {'initial sugar': +1},
                                 'residual sugar': {'initial sugar': +1}
                                 },
                         dominance=dominance,
                         infer_controls={
                             'alcohol': Dominance.MAX,
                             'quality': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class WineQualityWhite_2var_neg(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10_000,
                 seed: int = 42):
        super().__init__(file_name='wine_quality_white',
                         control=['volatile acidity','residual sugar','fixed acidity','total sulfur dioxide','density','pH','sulphates','quality','free sulfur dioxide','citric acid'],
                         preference=['alcohol','chlorides'],
                         effect={'quality': {'chlorides': -1, 'total sulfur dioxide': 1, 'pH': -1, 'alcohol': 0},
                                 'total sulfur dioxide': {'free sulfur dioxide': 1, 'reacted sulfur dioxide': 1},
                                 'pH': {'volatile acidity': -1, 'fixed acidity': -1},
                                 'fixed acidity': {'citric acid': +1},
                                 'alcohol': {'initial sugar': +1},
                                 'residual sugar': {'initial sugar': +1}
                                 },
                         dominance=dominance,
                         infer_controls={
                             'alcohol': Dominance.MAX,
                             'chlorides': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class WineQualityWhite_3var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10_000,
                 seed: int = 42):
        super().__init__(file_name='wine_quality_white',
                         control=['volatile acidity','residual sugar','fixed acidity','total sulfur dioxide','density','pH','sulphates','free sulfur dioxide','citric acid'],
                         preference=['alcohol','chlorides','quality'],
                         effect={'quality': {'chlorides': -1, 'total sulfur dioxide': 1, 'pH': -1, 'alcohol': 0},
                                 'total sulfur dioxide': {'free sulfur dioxide': 1, 'reacted sulfur dioxide': 1},
                                 'pH': {'volatile acidity': -1, 'fixed acidity': -1},
                                 'fixed acidity': {'citric acid': +1},
                                 'alcohol': {'initial sugar': +1},
                                 'residual sugar': {'initial sugar': +1}
                                 },
                         dominance=dominance,
                         infer_controls={
                             'alcohol': Dominance.MAX,
                             'chlorides': Dominance.MAX,
                             'quality': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class WineQualityWhite_4var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10_000,
                 seed: int = 42):
        super().__init__(file_name='wine_quality_white',
                         control=['volatile acidity','residual sugar','fixed acidity','total sulfur dioxide','pH','sulphates','free sulfur dioxide','citric acid'],
                         preference=['alcohol','chlorides','quality','density'],
                         effect={'quality': {'chlorides': -1, 'total sulfur dioxide': 1, 'pH': -1, 'alcohol': 0},
                                 'total sulfur dioxide': {'free sulfur dioxide': 1, 'reacted sulfur dioxide': 1},
                                 'pH': {'volatile acidity': -1, 'fixed acidity': -1},
                                 'fixed acidity': {'citric acid': +1},
                                 'alcohol': {'initial sugar': +1},
                                 'residual sugar': {'initial sugar': +1}
                                 },
                         dominance=dominance,
                         infer_controls={
                             'alcohol': Dominance.MAX,
                             'chlorides': Dominance.MAX,
                             'quality': Dominance.MAX,
                             'density': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)


class WineQualityWhite_5var_mixed(RealDataset):

    def __init__(self,
                 dominance: dict[str, Dominance] = None,
                 size: int = 10_000,
                 seed: int = 42):
        super().__init__(file_name='wine_quality_white',
                         control=['volatile acidity','residual sugar','fixed acidity','total sulfur dioxide','pH','sulphates','free sulfur dioxide'],
                         preference=['alcohol','chlorides','quality','density','citric acid'],
                         effect={'quality': {'chlorides': -1, 'total sulfur dioxide': 1, 'pH': -1, 'alcohol': 0},
                                 'total sulfur dioxide': {'free sulfur dioxide': 1, 'reacted sulfur dioxide': 1},
                                 'pH': {'volatile acidity': -1, 'fixed acidity': -1},
                                 'fixed acidity': {'citric acid': +1},
                                 'alcohol': {'initial sugar': +1},
                                 'residual sugar': {'initial sugar': +1}
                                 },
                         dominance=dominance,
                         infer_controls={
                             'alcohol': Dominance.MAX,
                             'chlorides': Dominance.MAX,
                             'quality': Dominance.MAX,
                             'density': Dominance.MAX,
                             'citric acid': Dominance.MAX
                         },
                         size=size,
                         seed=seed,
                         provided=False)

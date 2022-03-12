from pathlib import Path
import numpy as np
import scipy.constants as pc
import matplotlib.pyplot as plt


class Emitter(object):
    
    def __init__(self,
                 distance=230.0,  # port-to-beam distance (cm)
                 aperture=8.0,  # port aperture diameter (cm)
                 ):

        self.distance = distance
        self.aperture = aperture
        self.NA = (self.aperture/2) / self.distance

        self.spot_diameter = np.linspace(1, 2)

        print('Emitter specifications')
        print(f'  Port-beam distance: {self.distance:.1f} cm')
        print(f'  Aperture diameter: {self.aperture:.1f} cm')
        print(f'  Numerical aperture: {self.NA:.3g}')

    def calc_etendue(self):
        emitter_area = np.pi * (self.spot_diameter/2)**2 * 1e2  # mm^2
        etendue = np.pi * emitter_area * self.NA**2  # mm^2-ster
        return etendue
    
    def plot_etendue(self, save=False):
        etendue = self.calc_etendue()
        plt.figure(figsize=(4,3))
        plt.plot(self.spot_diameter, etendue)
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('Etendue (mm**2-ster)')
        plt.title(f'{self.aperture:.1f} cm dia. aper. | {self.distance:.0f} cm distance')
        plt.tight_layout()

        if save:
            fname = Path('plots') / 'emitter_etendue.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)


class Diode(object):
    
    def __init__(self,
                 t_op=-20.,  # operation temp (C)
                 t_ref=20.,  # reference temp (C)
                 dark_current_ref=10e-12,  # dark current @ t_ref (A)
                 responsivity=0.47,  # diode response (A/W)
                 bandwidth=1.e6,  # detector bandwidth (Hz)
                 r_feedback=100.e6,  # feedback resistor (Ohm)
                 distance=230.0,  # port-to-beam distance (cm)
                 aperture=8.0,  # port aperture diameter (cm)
                 ):
        
        self.emitter = Emitter(distance=distance, aperture=aperture)
        
        self.t_op = t_op
        self.t_ref = t_ref
        self.dark_current_ref = dark_current_ref
        self.responsivity = responsivity
        self.bandwidth = bandwidth
        self.r_feedback = r_feedback
        
        print('Diode specificaitons')
        print(f'  Operation temp: {self.t_op:.1f} C')
        print(f'  Reference temp: {self.t_ref:.1f} C')
        print(f'  Reference dark current: {self.dark_current_ref*1e9:.3f} nA')
        print(f'  Responsivity: {self.responsivity:.2f} A/W')
        print(f'  Bandwidth: {self.bandwidth/1e6:.1f} MHz')
        print(f'  Feedback resistor: {self.r_feedback/1e6:.1f} MOhm')

    def plot_diode_response(self, save=False):
        photon_wavelength = 656e-9  # wavelength (m)
        photon_energy = pc.h * pc.c / photon_wavelength  # energy per photon (J)
    
        self.spot_diameter = np.linspace(1, 2)
        etendue = self.emitter.calc_etendue()

        self.radiance = np.linspace(0.4, 1.2)*1e18  # photons/s/m**2/ster

        # incident power at colleciton aperture
        p_incident = photon_energy * np.matmul(self.radiance.reshape(-1,1), 
                                               etendue.reshape(1,-1)) / 1e6  # W
        
        transmission = 0.5
        # 50% transmission
        p_diode = transmission * p_incident
        # diode photocurrent
        i_photo = self.responsivity * p_diode
        # diode dark current
        i_dark = self.dark_current_ref * 2**((self.t_op-self.t_ref)/10)
        print(f'  Dark current @ T_op: {i_dark*1e9:.4f} nA')
        # diode thermal noise from feedback resistor
        isq_thermal = 4*pc.k*(self.t_op+273)/self.r_feedback * self.bandwidth
        print(f'  Feedback resistor thermal noise: {np.sqrt(isq_thermal)*1e9:.3f} nA')
        # diode shot noise from current
        isq_shot = 2 * pc.e * (i_photo+i_dark) * self.bandwidth
        # net noise
        i_noise = np.sqrt(isq_thermal + isq_shot)
        # signal-to-noise in photocurrent
        self.i_snr = i_photo / i_noise

        ncols, nrows = 3, 2
        fig, axes = plt.subplots(ncols=ncols,
                                 nrows=nrows, 
                                 figsize=(ncols*4,nrows*3))
        
        plt.sca(axes.flat[0])
        plt.contourf(self.spot_diameter, 
                     self.radiance/1e18, 
                     p_diode*1e9,
                     levels=4,
                     cmap='viridis')
        plt.colorbar(label='Flux (nW)')
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('self.Radiance (1e18 ph/s/m**2/st)')
        plt.title(f'Flux on diode ({transmission*1e2:.0f}% trans.)')
        
        plt.sca(axes.flat[1])
        plt.contourf(self.spot_diameter, 
                     self.radiance/1e18, 
                     i_photo*1e9,
                     levels=4,
                     cmap='viridis')
        plt.colorbar(label='Photocurrent (nA)')
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('self.Radiance (1e18 ph/s/m**2/st)')
        plt.title('Photocurrent')
        
        plt.sca(axes.flat[2])
        plt.contourf(self.spot_diameter, 
                     self.radiance/1e18, 
                     i_noise*1e9,
                     levels=4,
                     cmap='viridis')
        plt.colorbar(label='Noise current (nA)')
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('self.Radiance (1e18 ph/s/m**2/st)')
        plt.title('Noise current (shot+therm)')
        
        plt.sca(axes.flat[3])
        plt.contourf(self.spot_diameter, 
                     self.radiance/1e18, 
                     np.sqrt(isq_shot)*1e9,
                     levels=4,
                     cmap='viridis')
        plt.colorbar(label='Shot noise (nA)')
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('self.Radiance (1e18 ph/s/m**2/st)')
        plt.title('Shot noise')
        
        plt.sca(axes.flat[4])
        plt.contourf(self.spot_diameter, 
                     self.radiance/1e18, 
                     self.i_snr,
                     levels=4,
                     cmap='viridis')
        plt.colorbar(label='SNR')
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('self.Radiance (1e18 ph/s/m**2/st)')
        plt.title('Signal-to-noise ratio')
        
        plt.tight_layout()

        if save:
            fname = Path('plots') / 'diode_response.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
            
    def plot_min_detect_fluctuation(self, 
                                    di_dn_factor=0.55,
                                    save=False):
        deli_i = 1/self.i_snr
        plt.figure(figsize=(4,3))
        plt.contourf(self.spot_diameter, 
                     self.radiance/1e18, 
                     deli_i*1e2 / di_dn_factor,
                     levels=np.arange(0.25,1.51,0.25),
                     cmap='viridis')
        plt.colorbar(label='Minimum delta-n/n (%)')
        plt.xlabel('Spot diameter (cm)')
        plt.ylabel('self.Radiance (1e18 ph/s/m**2/st)')
        plt.title(f'(dI/I)/(dn/n) = {di_dn_factor:.2f}')
        plt.tight_layout()

        if save:
            fname = f'min_detect_fluctuation_f{di_dn_factor*1e2:.0f}.pdf'
            fname = Path('plots') / fname
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)


if __name__=='__main__':
    plt.close('all')

    diode = Diode()
    diode.emitter.plot_etendue(save=True)
    diode.plot_diode_response(save=True)
    diode.plot_min_detect_fluctuation(save=True)

    plt.show()
import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

try:
    from .utilities import vmec_connection
    from .beams import HeatingBeam
except ImportError:
    from w7x_bes_tools.utilities import vmec_connection
    from w7x_bes_tools.beams import HeatingBeam


# vmec connection
vmec = vmec_connection.connection()
Points3D = vmec.type_factory('ns1').Points3D


class Sightline(object):
    
    # vmec Fourier coefficients
    pmodes = None
    tmodes = None
    rcoeff_interp = None
    zcoeff_interp = None
    
    def __init__(self, 
                 beam=None, 
                 port=None,
                 r_obs=5.88, 
                 z_obs=-0.4, 
                 eq_tag=None):
        if not beam:
            beam = HeatingBeam(source=2)
        if eq_tag is not None and eq_tag!=beam.eq_tag:
            beam.set_eq(eq_tag=eq_tag)
            self.pmodes = self.tmodes = None
            self.rcoeff_interp = self.zcoeff_interp = None
        self.eq_tag = beam.eq_tag
        if not vmec or not Points3D:
            raise ValueError
        self.port = port
        self.beam = beam
        # calc dist from source to axis point at R=r_obs
        rxy = beam.r_hat[0:2]
        srcxy = beam.source[0:2]
        quad_a = np.linalg.norm(rxy)**2
        quad_b = 2*np.inner(rxy,srcxy)
        quad_c = np.linalg.norm(srcxy)**2-r_obs**2
        dist = (-quad_b - np.sqrt(quad_b**2-4*quad_a*quad_c)) / (2*quad_a)
        if z_obs is None:
            # if None, take z_obs to be on beam axis
            target = beam.source + dist*beam.r_hat
            z_obs = target[2]
        else:
            def fun(x):
                dr,dt = x
                f1 = (beam.source[0]+dr*beam.r_hat[0]+dt*beam.t_hat[0])**2 \
                     + (beam.source[1]+dr*beam.r_hat[1]+dt*beam.t_hat[1])**2 \
                     - r_obs**2
                f2 = beam.source[2] + dr*beam.r_hat[2] + dt*beam.t_hat[2] - z_obs
                return [f1, f2]
            solution = optimize.root(fun, [dist,0], jac=False, options={})
            if not solution.success:
                print(solution.message)
                raise RuntimeError(solution.message)
            dr,dt = solution.x
            # beam target coordinates
            target = beam.source + dr*beam.r_hat + dt*beam.t_hat
        self.r_obs = r_obs
        self.z_obs = z_obs
        # port coordinates
        port_xyz = beam.ports[self.port]
        # vector from port to beam target
        sightline = target - port_xyz
        obs_distance = np.linalg.norm(sightline)
        # sightline unit vector
        uvector_sl = sightline / obs_distance
        # step interval along sightline
        step_sl= 0.02
        # distance array along sightline (near beam axis)
        dist_sl = np.arange(-0.4, 0.4*(1+1e-4), step_sl) + obs_distance
        # x,y,z coords along sightline (near beam axis)
        xyz_sl = port_xyz.reshape(3,1) + np.outer(uvector_sl, dist_sl)
        # r_eff along sightline
        vmec_reff = vmec.service.getReff(self.eq_tag, 
                                         Points3D(*xyz_sl.tolist()))
        reff_sl = np.array(vmec_reff)
        # test for inside LCFS
        inplasma = np.isfinite(reff_sl)
        if np.count_nonzero(inplasma) == 0:
            raise ValueError
        # remove sightline points outside LCFS
        xyz_sl = xyz_sl[:,inplasma]
        dist_sl = dist_sl[inplasma]
        reff_sl = reff_sl[inplasma]
        ngrid = dist_sl.size
        self.imax = np.argmin(np.abs(dist_sl-obs_distance))
        vmec_bvector = vmec.service.magneticField(self.eq_tag, 
                                                  Points3D(*xyz_sl.tolist()))
        bvec_xyz_sl = np.array([vmec_bvector.x1, vmec_bvector.x2, vmec_bvector.x3])
        bnorm_sl = np.linalg.norm(bvec_xyz_sl,axis=0)
        buvec_sl = bvec_xyz_sl / np.tile(bnorm_sl.reshape(1,ngrid), (3,1))
        assert(np.allclose(np.linalg.norm(buvec_sl, axis=0), 1))
        # psi
        rpz_sl = beam.xyz_to_rpz(xyz_sl)
        r_sl = rpz_sl[0,:]
        z_sl = xyz_sl[2,:]
        vmec_stp = vmec.service.toVMECCoordinates(self.eq_tag, 
                                                  Points3D(*rpz_sl.tolist()),
                                                  1e-4)
        psi_sl = np.array(vmec_stp.x1) # norm. flux surface
        theta_sl = np.array(vmec_stp.x2) # poloidal angle [rad]
        theta_sl[theta_sl>np.pi] = theta_sl[theta_sl>np.pi] - 2*np.pi
        phi_sl = np.array(vmec_stp.x3) # toroidal angle [rad]
        # calc angle wrt B vector along sightline
        dots = np.sum(np.tile(uvector_sl.reshape(3,1), 
                              (1,ngrid)) * buvec_sl, axis=0)
        bangle_sl = np.arccos(dots)
        bangle_sl[np.isnan(bangle_sl)] = np.pi/2
        bangle_sl = np.pi/2-np.abs(bangle_sl-np.pi/2)
        # print('  R={:.3f} Z={:.3f} psi={:.3f} |B|={:.3f} l_plasma={:.3f}'.format(
        #     r_obs, z_obs, psi_sl[self.imax], bnorm_sl[self.imax], dist_sl[-1]-dist_sl[0]))
        # calc beam intensity along sightline
        beam_intensity_sl = np.empty(ngrid)
        for i in np.arange(ngrid):
            beam_intensity_sl[i] = beam.point_intensity(x=xyz_sl[0,i],
                                                        y=xyz_sl[1,i],
                                                        z=xyz_sl[2,i])
        # normalize intensity along sightline
        beam_intensity_sl = beam_intensity_sl / beam_intensity_sl.max()
        # fourier components
        if self.rcoeff_interp is None:
            self.vmec_coeff()
        u_sl = theta_sl
        dphidv = 1/5
        v_sl = phi_sl / dphidv
        def calc_nhat(s,u,v, rin, zin):
            rcoeff_local = self.rcoeff_interp(s)
            zcoeff_local = self.zcoeff_interp(s)
            phi = v * dphidv
            parray = np.broadcast_to(self.pmodes, rcoeff_local.shape)
            tarray = np.broadcast_to(self.tmodes, rcoeff_local.shape)
            operand = parray*u - tarray*v
            cosmn = np.cos(operand)
            sinmn = np.sin(operand)
            r = np.sum(rcoeff_local * cosmn)
            z = np.sum(zcoeff_local * sinmn)
            if not np.allclose(r, rin, rtol=5e-4):
                print(r, rin, s)
                assert(False)
            if not np.allclose(z, zin, rtol=5e-4):
                print(z, zin, s)
                assert(False)
            dcosdu = -parray * sinmn
            dcosdv = tarray * sinmn
            dsindu = parray * cosmn
            dsindv = -tarray * cosmn
            eu = np.array([np.sum(rcoeff_local * dcosdu) * np.cos(phi),
                           np.sum(rcoeff_local * dcosdu) * np.sin(phi),
                           np.sum(zcoeff_local * dsindu)])
            eu = eu / np.linalg.norm(eu)
            evx = np.sum(rcoeff_local*dcosdv)*np.cos(phi) - r*dphidv*np.sin(phi)
            evy = np.sum(rcoeff_local*dcosdv)*np.sin(phi) + r*dphidv*np.cos(phi)
            evz = np.sum(zcoeff_local*dsindv)
            ev = np.array([evx, evy, evz])
            ev = ev / np.linalg.norm(ev)
            nvec = -np.cross(eu,ev)
            nvec = nvec / np.linalg.norm(nvec)
            nx = (-(np.sum(rcoeff_local*dcosdu)*np.sum(zcoeff_local*dsindv) -
                  np.sum(rcoeff_local*dcosdv)*np.sum(zcoeff_local*dsindu))*np.sin(phi) + 
                  r*dphidv*np.sum(zcoeff_local * dsindu)*np.cos(phi))
            ny = ((np.sum(rcoeff_local*dcosdu)*np.sum(zcoeff_local*dsindv) -
                  np.sum(rcoeff_local*dcosdv)*np.sum(zcoeff_local*dsindu))*np.cos(phi) +
                  r*dphidv*np.sum(zcoeff_local * dsindu)*np.sin(phi))
            nz = -r*dphidv*np.sum(rcoeff_local*dcosdu)
            nvec2 = np.array([nx,ny,nz])
            nvec2 = nvec2 / np.linalg.norm(nvec2)
            assert(np.allclose(nvec,nvec2))
            return nvec
        self.nhat = np.empty(buvec_sl.shape)
        self.bihat = np.empty(buvec_sl.shape)
        self.cosn = np.empty(ngrid)
        self.cosbi = np.empty(ngrid)
        for igrid in np.arange(ngrid):
            self.nhat[:,igrid] = calc_nhat(psi_sl[igrid], u_sl[igrid], v_sl[igrid], 
                                      r_sl[igrid], z_sl[igrid])
            assert(np.allclose(np.sum(self.nhat[:,igrid]*buvec_sl[:,igrid])+1, 1,
                               rtol=5e-4))
            self.bihat[:,igrid] = np.cross(self.nhat[:,igrid], buvec_sl[:,igrid])
            assert(np.allclose(np.linalg.norm(self.bihat[:,igrid]), 1))
            self.bihat[:,igrid]  = self.bihat[:,igrid] / np.linalg.norm(self.bihat[:,igrid])
            self.cosn[igrid] = np.sum(uvector_sl*self.nhat[:,igrid])
            self.cosbi[igrid] = np.sum(uvector_sl*self.bihat[:,igrid])
        self.slhat = uvector_sl
        self.distance = dist_sl
        self.step = step_sl
        self.xyz = xyz_sl
        self.reff = reff_sl
        self.r = r_sl
        self.z = z_sl
        self.r_obs = r_obs
        self.z_obs = z_obs
        self.r_avg = np.average(r_sl, weights=beam_intensity_sl)
        self.z_avg = np.average(z_sl, weights=beam_intensity_sl)
        self.psinorm = psi_sl
        self.phi = phi_sl
        self.theta = theta_sl
        self.bangle = bangle_sl * 180 / np.pi
        self.intensity = beam_intensity_sl
        self.imaxbeam = beam_intensity_sl.argmax()
        self.bhat = buvec_sl
        self.bnorm = bnorm_sl
        self.norm_half_excursion = np.abs(np.sum(self.cosn*self.step*self.intensity))/2
        self.binorm_half_excursion = np.abs(np.sum(self.cosbi*self.step*self.intensity))/2
        nhat = self.nhat[:,self.imax]
        bihat = self.bihat[:,self.imax]
        nhat_r = np.linalg.norm(nhat[0:2])
        nhat_z = nhat[2]
        bihat_r = np.linalg.norm(bihat[0:2])
        bihat_z = bihat[2]
        self.rseq = [self.r_avg + self.binorm_half_excursion*bihat_r,
                     self.r_avg + self.norm_half_excursion*nhat_r,
                     self.r_avg - self.binorm_half_excursion*bihat_r,
                     self.r_avg - self.norm_half_excursion*nhat_r,
                     # self.r_avg + self.binorm_half_excursion*bihat_r,
                     ]
        self.zseq = [self.z_avg + self.binorm_half_excursion*bihat_z,
                     self.z_avg + self.norm_half_excursion*nhat_z,
                     self.z_avg - self.binorm_half_excursion*bihat_z,
                     self.z_avg - self.norm_half_excursion*nhat_z,
                     # self.z_avg + self.binorm_half_excursion*bihat_z,
                     ]
        
    def vmec_coeff(self):
        vmec_rcoscoeff = vmec.service.getFourierCoefficients(self.eq_tag, 'RCos')
        vmec_zsincoeff = vmec.service.getFourierCoefficients(self.eq_tag, 'ZSin')
        nrad = vmec_rcoscoeff.numRadialPoints
        rarray = np.linspace(0,1,nrad) # uniform psinorm grid
        self.pmodes = np.array(vmec_rcoscoeff.poloidalModeNumbers, dtype=np.int).reshape([-1,1])
        npol = self.pmodes.size
        self.tmodes = np.array(vmec_rcoscoeff.toroidalModeNumbers, dtype=np.int).reshape([1,-1])
        ntor = self.tmodes.size
        rcoeff = np.empty([nrad,npol,ntor])
        zcoeff = np.empty([nrad,npol,ntor])
        for k in np.arange(nrad):
            for m in np.arange(npol):
                for n in np.arange(ntor):
                    icoeff = (m * ntor + n) * nrad + k
                    rcoeff[k,m,n] = vmec_rcoscoeff.coefficients[icoeff]
                    zcoeff[k,m,n] = vmec_zsincoeff.coefficients[icoeff]
        self.rcoeff_interp = CubicSpline(rarray, rcoeff, axis=0, extrapolate=False)
        self.zcoeff_interp = CubicSpline(rarray, zcoeff, axis=0, extrapolate=False)
        
    def plot_sightline(self, save=False):
        # plot quantities along sl
        plt.figure(figsize=(12,8.25))
        plotdata = [[self.r, 'R-major (m)'],
                    [self.z, 'Z (m)'],
                    [self.bangle, 'Angle wrt B (deg)'],
                    [self.psinorm, 'Psi-norm'], 
                    [self.cosn, 'sl*nhat'],
                    [self.cosbi, 'sl*bihat'],
                    ]
        for i,pdata in enumerate(plotdata):
            plt.subplot(3,3,i+1)
            xdata = self.distance
            ydata = pdata[0]
            plt.xlim(xdata.min(), xdata.max())
            plt.ylim(ydata.min()-np.abs(ydata.min())*0.1, 
                      ydata.max()+np.abs(ydata.max())*0.1)
            points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, norm=plt.Normalize(0,1),
                                cmap='viridis_r')
            lc.set_array(self.intensity)
            lc.set_linewidth(3)
            line = plt.gca().add_collection(lc)
            w_avg, w_sd = self.beam.weighted_avg(ydata, self.intensity)
            plt.errorbar(xdata[self.imax], w_avg, yerr=w_sd, 
                         color='r', capsize=3)
            plt.colorbar(line, ax=plt.gca())
            plt.xlabel('Dist. on {} sightline (m)'.format(self.port))
            plt.ylabel(pdata[1])
            plt.title('{} | {} | {}'.format(self.eq_tag, self.port, self.beam.name))
            if np.array_equal(ydata, self.bangle):
                plt.ylim(0,10)
            plt.annotate('beam-wtd {}'.format(pdata[1]), (0.05,0.9), 
                          xycoords='axes fraction')
            plt.annotate('{:.3g} +/- {:.2g}'.format(w_avg, w_sd), (0.05,0.8), 
                          xycoords='axes fraction')
            if np.array_equal(ydata, self.cosn) or np.array_equal(ydata, self.cosbi):
                w_int = np.abs(np.sum(ydata * self.step * self.intensity))/2
                plt.annotate('beam-wtd-sum/2: {:.3g} cm'.format(w_int*1e2), (0.05,0.7), 
                          xycoords='axes fraction')
        # add measurement markers to beam profiles
        ax1, ax2 = self.beam.plot_beam_plane(self.port, sp1=337, sp2=338)
        plt.sca(ax1)
        plt.plot(self.rseq, self.zseq, color='r')
        plt.sca(ax2)
        plt.plot(self.rseq, self.zseq, color='b')
        plt.tight_layout()
        if save:
            fname = self.beam._plots_dir / \
                        'S{:d}_view_from_{}_R{:.0f}_Z{:.0f}.pdf'.format(
                            self.beam.injector, 
                            self.port, 
                            np.round(self.r_obs*1e2), 
                            np.round(self.z_obs*1e2))
            plt.savefig(fname.as_posix(), transparent=True)


if __name__=='__main__':
    plt.close('all')
    
    p2 = HeatingBeam(source=7, eq_tag='w7x_ref_29')

    s = Sightline(p2, port='A21-hihi', r_obs=6.03, z_obs=-0.16)
    s = Sightline(p2, port='A21-hihi', r_obs=5.84, z_obs=-0.52)
    s.plot_sightline(save=True)

    plt.show()
    

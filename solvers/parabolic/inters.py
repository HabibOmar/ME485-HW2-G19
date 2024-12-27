# -*- coding: utf-8 -*-
from solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from backends.types import Kernel, NullKernel
from solvers.parabolic.bcs import get_bc

from utils.np import npeval
from utils.nb import dot
import functools as fc
import numpy as np
import re

#-------------------------------------------------------------------------------#    
class ParabolicIntInters(BaseIntInters):
    def construct_kernels(self, elemap, impl_op):
        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

        self._correction = self.cfg.get('solver', 'correction', 'minimum')
       
        # Array for gradient at face
        self._gradf  = gradf   = np.empty((self.ndims, self.nvars, self.nfpts))
        self._weight = weight  = np.empty((self.nfpts))

        if self.order > 1:
            self.compute_delu = Kernel(self._make_delu(), *fpts)
            dxf = [cell.dxf for cell in elemap.values()]        
            self.compute_weight(dxf)
        else:
            self.compute_delu = NullKernel

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at_face = Kernel(
                                    self._make_grad_at_face(nele), gradf, *fpts, *dfpts)

        muf = np.empty(self.nfpts)
        self.compute_flux = Kernel(self._make_flux(nele), muf, gradf, *fpts)


#-------------------------------------------------------------------------------#    
    def compute_weight(self, dxf):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        for idx in range(self.nfpts):
            lti, lfi, lei = lt[idx], lf[idx], le[idx]
            rti, rfi, rei = rt[idx], rf[idx], re[idx]

            dl = 0.0
            dr = 0.0
            for jdx in range(ndims):
                xjl = dxf[lti][lfi, jdx, lei]
                xjr = dxf[rti][rfi, jdx, rei]
                dl += xjl*xjl
                dr += xjr*xjr

            dl  = np.sqrt(dl)
            dr  = np.sqrt(dr)

            self._weight[idx] = dr/(dl+dr)

#-------------------------------------------------------------------------------#    
    def _make_flux(self, nele):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm  # nf.shape (ndims, nfpts), sf.shape (nfpts, )
        # Inverse distance between the elements
        inv_ef    = self._rcp_dx  # inv_ef.shape (nfpts, )
        
        # Unit vector connecting cell centers 
        ef = self._dx_adj * inv_ef

        # Correction method to be used
        correction = self._correction

        # Compiler arguments
        array = self.be.local_array()
        
        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()

        def comm_flux(i_begin, i_end, muf, gradf, *uf):
            # Parse element views (fpts, grad)
            du = uf[:nele]

            for idx in range(i_begin, i_end):
                #*************************# 
                # Complete function

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                # Compute viscosity
                muf[idx] = compute_mu(idx)  # Constant viscosity "Diffusion Coefficient"


                # Face area and unit normal vector
                S_f = sf[idx]
                n_f = nf[:, idx]

                # Unit vector along connecting line between cell centers and inverse distance
                e_f = ef[:, idx]
                inv_e_f = inv_ef[idx]

                fn = np.zeros(nfvars)

                for jdx in range(nfvars):

                    grad_face = gradf[:, jdx, idx] # Gradients at face
                    dui = du[lti][lfi, jdx, lei]  # difference q_r - q_l


                    # Minimum correction:    E_f = (e·S_f) e = (S_f * (e·n_f)) e
                    # Orthogonal correction: E_f = S_f · e
                    # Over-relaxed:          E_f = ((S_f)/(e·n_f)) e

                    S_f_vec = S_f * n_f 

                    if correction == 'minimum':
                        E_f = S_f * (np.dot(e_f, n_f)) * e_f  # = S_f(e_f·n_f) e_f

                    elif correction == 'orthogonal':
                        E_f = S_f * e_f # = S_f·e
 
                    elif correction == 'over_relaxed':
                        E_f = (S_f/np.dot(e_f, n_f)) * e_f   # = (S_f/(e·n_f)) e_f
                        
                    # non-orthogonal like contribution: T_f = S_f - E_f ;  ∇ϕ_f · T_f
                    T_f = S_f_vec - E_f
                    grad_T_f = np.dot(grad_face, T_f)

                    #orthogonal like contribution: ||E_f|| * ϕc - ϕf / dcf
                    E_f_mag = np.linalg.norm(E_f)
                    grad_E_f = E_f_mag * (dui*inv_e_f)                    

                
                    # Diffusion flux with correction: F = -mu (grad_T_f + grad_E_f)
                    fn[jdx] = -muf[idx] * (grad_T_f + grad_E_f)
                #*************************#              

                    uf[lti][lfi, jdx, lei] =  fn[jdx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]

        return self.be.make_loop(self.nfpts, comm_flux)


#-------------------------------------------------------------------------------#    
    def _make_grad_at_face(self, nele):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx    
        # Inverse distance between the cell center
        weight    = self._weight # weight.shape (nfpts, )
        # Stack-allocated array
        array = self.be.local_array()

        def grad_at(i_begin, i_end, gradf, *uf):  #gradf.shape (ndims, nvars, nfpts)
            # Parse element views (fpts, grad)
            du    = uf[:nele]  # shape (nele, nvars, nface)
            gradu = uf[nele:]  # shape (ndims, nvars, nele)

            for idx in range(i_begin, i_end):
            #*************************#  
            # Complete function
                lti, lei = lt[idx], le[idx]
                rti, rei = rt[idx], re[idx]

                # Weight for gradient averaging
                w = weight[idx]
                # print(w)
                w_l = w
                w_r = 1.0 - w_l

            # Weighted average o grfadients:
            # (∇q)_face = w_l*(∇q)_left + w-*(∇q)_right
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gl = gradu[lti][kdx, jdx, lei]
                        gr = gradu[rti][kdx, jdx, rei]
                        grad_face = w_l*gl +  w_r*gr
                        gradf[kdx, jdx, idx] = grad_face            
            #*************************# 

        return self.be.make_loop(self.nfpts, grad_at)

#-------------------------------------------------------------------------------#    
    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = uf[rti][rfi, jdx, rei]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] =  du
                    uf[rti][rfi, jdx, rei] = -du

        return self.be.make_loop(self.nfpts, compute_delu)

#-------------------------------------------------------------------------------#    
class ParabolicBCInters(BaseBCInters):
    _get_bc = get_bc
    def construct_bc(self):
        # Parse BC function name
        bcf = re.sub('-', '_', self.name)

        # Constants for BC function
        if self._reqs:
            bcsect = 'soln-bcs-{}'.format(self.bctype)
            bcc = {k: npeval(self.cfg.getexpr(bcsect, k, self._const))
                   for k in self._reqs}
        else:
            bcc = {}

        bcc['ndims'], bcc['nvars'], bcc['nfvars'] = self.ndims, self.nvars, self.nfvars

        

        bcc.update(self._const)
        
        # Get bc from `bcs.py` and compile them
        self.bc = self._get_bc(self.be, bcf, bcc)

#-------------------------------------------------------------------------------#    
    def construct_kernels(self, elemap, impl_op):
        self.construct_bc()

        self._correction = self.cfg.get('solver', 'correction', 'minimum')
        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

       # Gradient at face
        self._gradf = gradf = np.empty((self.ndims, self.nvars, self.nfpts))

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(self._make_delu(), *fpts)

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at_face = Kernel(
            self._make_grad_at_face(nele), gradf, *fpts, *dfpts
        )
  
        # Save viscosity on face (for implicit operator)
        muf = np.empty(self.nfpts)

        # Kernel to compute flux
        self.compute_flux = Kernel(self._make_flux(nele), muf, gradf, *fpts)

#-------------------------------------------------------------------------------#    
    def _make_flux(self, nele):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Mangitude and direction of the connecting vector
        inv_ef = self._rcp_dx
        ef = self._dx_adj * inv_ef
        # avec = self._vec_snorm/np.einsum('ij,ij->j', ef, self._vec_snorm)

        correction = self._correction

        # Compiler arguments
        array = self.be.local_array()
       
        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()

        # Get bc function 
        bc = self.bc

        def comm_flux(i_begin, i_end, muf, gradf, *uf):
            # Parse element views (fpts, grad)
            du    = uf[:nele]
            for idx in range(i_begin, i_end):
                #*************************# 
                # Complete function

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                # Compute viscosity
                muf[idx] = compute_mu(idx)  # Constant viscosity "Diffusion Coefficient"


                # Face area and unit normal vector
                S_f = sf[idx]
                n_f = nf[:, idx]

                # Unit vector along connecting line between cell centers and inverse distance
                e_f = ef[:, idx]
                inv_e_f = inv_ef[idx]

                fn = np.zeros(nfvars)

                for jdx in range(nfvars):

                    grad_face = gradf[:, jdx, idx] # Gradients at face
                    dui = du[lti][lfi, jdx, lei]  # difference q_r - q_l


                    # Minimum correction:    E_f = (e·S_f) e = (S_f * (e·n_f)) e
                    # Orthogonal correction: E_f = S_f · e
                    # Over-relaxed:          E_f = ((S_f)/(e·n_f)) e

                    S_f_vec = S_f * n_f 

                    if correction == 'minimum':
                        E_f = S_f * (np.dot(e_f, n_f)) * e_f  # = S_f(e·n_f) e_f

                    elif correction == 'orthogonal':
                        E_f = S_f * e_f # = S_f·e
 
                    elif correction == 'over_relaxed':
                        E_f = (S_f/np.dot(e_f, n_f)) * e_f   # = (S_f/(e·n_f)) e_f
                        
                    # non-orthogonal like contribution: T_f = S_f - E_f ;  ∇ϕ_f · T_f
                    T_f = S_f_vec - E_f
                    grad_T_f = np.dot(grad_face, T_f)

                    #orthogonal like contribution: ||E_f|| * ϕc - ϕf / dcf
                    E_f_mag = np.linalg.norm(E_f)
                    grad_E_f = E_f_mag * (dui*inv_e_f)                    

                
                    # Diffusion flux with correction: F = -mu (grad_T_f + grad_E_f)
                    fn[jdx] = -muf[idx] * (grad_T_f + grad_E_f)
                    
                    # Assigning
                    uf[lti][lfi, jdx, lei] =  fn[jdx]

                #*************************# 
                
        return self.be.make_loop(self.nfpts, comm_flux)

#-------------------------------------------------------------------------------#    
    def _make_grad_at_face(self, nele):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx

        # Mangitude and direction of the connecting vector
        inv_tf = self._rcp_dx
        tf = self._dx_adj * inv_tf
        avec = self._vec_snorm/np.einsum('ij,ij->j', tf, self._vec_snorm)

        # Stack-allocated array
        array = self.be.local_array()

        def grad_at(i_begin, i_end, gradf, *uf):
            # Parse element views (fpts, grad)
            du = uf[:nele]
            gradu = uf[nele:]

            for idx in range(i_begin, i_end):
               #*************************# 
               # Complete function
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    for kdx in range(ndims):
                        d_u = du[lti][lfi, jdx, lei]
                        gradf[kdx, jdx, idx] = d_u*inv_tf[idx]*avec[kdx, idx]
               #*************************# 

        return self.be.make_loop(self.nfpts, grad_at)

#-------------------------------------------------------------------------------#    
    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc
        array = self.be.local_array()

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                ur = array(nvars)
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                bc(ul, ur, nfi)
                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)


#-------------------------------------------------------------------------------#    
class ParabolicDrichletBCInters(ParabolicBCInters):
    name = 'drichlet'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        self._reqs = self.primevars


#-------------------------------------------------------------------------------#    
class ParabolicNeumannBCInters(ParabolicBCInters):
    name = 'neumann'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        self._reqs = self.primevars

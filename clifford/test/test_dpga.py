
import numpy as np
import pytest
from clifford.dpga import *

from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays as st_arrays


test_double = st.floats(-1E-2, 1E+2, allow_nan=False, allow_infinity=False, width=64)


class TestBasicDPGA:
    def test_non_orthogonal_metric(self):
        w_metric = np.array([
            [
                (a | b)[0]
                for a in wbasis
            ]
            for b in wbasis
        ])
        assert np.all(w_metric == np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]) / 2)

    def test_bivector_identities(self):
        """
        These come from section 2 of the paper:
        R(4, 4) As a Computational Framework for 3-Dimensional Computer Graphics
        by Ron Goldman and Stephen Mann
        """
        for wi in wlist:
            for wj in wlist:
                assert wi^wj == -wj*wi

        for wis in wslist:
            for wjs in wslist:
                assert wis^wjs == -wjs*wis

        for w in wbasis:
            assert w**2 == 0*w1

        for wi, wis in zip(wlist, wslist):
            assert wi*wis == 1 - wis*wi

    @settings(deadline=None)
    @given(p=st_arrays(np.double, 3, elements=test_double))
    def test_up_down(self, p):
        dpga_pnt = up(p)
        # This is a homogeneous point representation
        pnt_down = down(5.0*dpga_pnt)
        np.testing.assert_allclose(pnt_down, p)

    @settings(deadline=None)
    @given(tvec=st_arrays(np.double, 3, elements=test_double),
           pnt_vec=st_arrays(np.double, 3, elements=test_double))
    def test_translate(self, tvec, pnt_vec):
        wt = tvec[0]*w1 + tvec[1]*w2 + tvec[2]*w3
        biv = w0s*wt
        Rt = 1 - biv
        exp_result = np.e**(-biv)
        assert Rt == exp_result

        assert Rt * w0 * ~Rt == w0 + wt
        for wi in [w1, w2, w3]:
            assert Rt * wi * ~Rt == wi

        assert (Rt*~Rt) == 1 + 0*w1

        pnt = up(pnt_vec)
        res = Rt*pnt*~Rt
        desired_result = up(pnt_vec+tvec)

        assert up(pnt_vec) + wt == desired_result
        assert res == desired_result

    @settings(deadline=None)
    @given(mvec=st_arrays(np.double, 3, elements=test_double),
           nvec=st_arrays(np.double, 3, elements=test_double),
           pnt_vec=st_arrays(np.double, 3, elements=test_double))
    def test_rotate(self, mvec, nvec, pnt_vec):
        m = mvec[0] * w1 + mvec[1] * w2 + mvec[2] * w3
        n = nvec[0] * w1 + nvec[1] * w2 + nvec[2] * w3
        ms = mvec[0] * w1s + mvec[1] * w2s + mvec[2] * w3s
        ns = nvec[0] * w1s + nvec[1] * w2s + nvec[2] * w3s
        biv = 2*((ms^n) - (ns^m))
        Rt = np.e**(-biv)

        # Rotor should be unit
        np.testing.assert_allclose((Rt*~Rt).clean().value, (1 + 0*w1).value, atol=1E-3, rtol=1E-2)

        # The origin should be unaffected by rotation
        origin_transform = (Rt * w0 * ~Rt)
        np.testing.assert_allclose(origin_transform(1).value, w0.value, atol=1E-3, rtol=1E-2)
        np.testing.assert_allclose(origin_transform - origin_transform(1), 0, atol=1E-6)

        # Vectors orthogonal to the rotation should be unaffected by rotation
        vorthog = np.cross(mvec, nvec)
        uporthog = up(vorthog)
        np.testing.assert_allclose((Rt*uporthog*~Rt)(1).value, uporthog.value, atol=1E-3, rtol=1E-2)

        # Points should maintain their distance from the origin
        l = np.linalg.norm(pnt_vec)
        pnt = up(pnt_vec)
        lres = np.linalg.norm(down(Rt * pnt * ~Rt))
        np.testing.assert_allclose(l, lres, atol=1E-6)

    def test_line(self):
        rng = np.random.default_rng()  # can pass a seed here later
        for i in range(100):
            p1vec = rng.standard_normal(3)
            p2vec = rng.standard_normal(3)
            p1 = up(p1vec)
            p2 = up(p2vec)
            line_direc = p2vec - p1vec

            # Plucker line is the outer product of two points or the outer product
            # of a point and a free vector
            line = p1 ^ p2
            free_direc = line_direc[0]*w1 + line_direc[1]*w2 + line_direc[2]*w3
            line_alt = p1 ^ free_direc
            assert line_alt == line

            # The line should be the outer product null space
            lamb = rng.standard_normal()
            assert up(lamb*p1vec + (1 - lamb)*p2vec) ^ line == 0

            # Lines can be transformed with rotors
            tvec = p1vec
            wt = tvec[0] * w1 + tvec[1] * w2 + tvec[2] * w3
            Raxis = 1 - w0s * wt
            assert (~Raxis*line*Raxis) ^ w0 == 0

            # Lines are invariant to screw transformations about their axis
            axis = p1vec - p2vec
            rotation_biv = axis[0]*(e23 - e2b3b) + axis[1]*(e1b3b - e13) + axis[2]*(e12 - e1b2b)
            Rr = np.e**(-rng.standard_normal()*rotation_biv)
            Rt = 1 - w0s * rng.standard_normal()*(axis[0] * w1 + axis[1] * w2 + axis[2] * w3)
            np.testing.assert_allclose((Raxis*Rr*Rt*(~Raxis*line*Raxis)*~Rt*~Rr*~Raxis).value,
                                       line.value, rtol=1E-4, atol=1E-4)

            # A bivector line is invariant under its own exponential
            Rline = np.e ** line
            assert Rline*line*~Rline == line

            # The exponential of a line is a rotation about the line
            line_origin = (~Raxis*line*Raxis)
            RLineOrigin = np.e ** line_origin
            random_pnt = up(rng.standard_normal(3))
            np.testing.assert_allclose(np.linalg.norm(down(RLineOrigin*random_pnt*~RLineOrigin)),
                                       np.linalg.norm(down(random_pnt)), rtol=1E-3, atol=1E-4)
            np.testing.assert_allclose(down(RLineOrigin * (random_pnt + free_direc) * ~RLineOrigin),
                                       down(RLineOrigin * random_pnt * ~RLineOrigin) + line_direc,
                                       rtol=1E-3, atol=1E-4)
            np.testing.assert_allclose((Rline*~Rline).value, (1 + 0*w1).value, rtol=1E-4, atol=1E-4)

    def test_quadric(self):
        rng = np.random.default_rng()  # can pass a seed here later
        # Make a cone which passes through the origin
        # This is the construction from Transverse Approach paper
        quadric_coefs = [0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a, b, c, d, e, f, g, h, i, j = quadric_coefs
        quadric = 4 * a * (w0s ^ w0) + 4 * b * (w1s ^ w1) + 4 * c * (w2s ^ w2) + 4 * j * (w3s ^ w3) + \
                  2 * d * ((w0s ^ w1) + (w1s ^ w0)) + 2 * e * ((w0s ^ w2) + (w2s ^ w0)) + \
                  2 * f * ((w1s ^ w2) + (w2s ^ w1)) + 2 * g * ((w0s ^ w3) + (w3s ^ w0)) + \
                  2 * h * ((w1s ^ w3) + (w3s ^ w1)) + 2 * i * ((w2s ^ w3) + (w3s ^ w2))

        # The quadrics do not form an OPNS
        assert quadric ^ w0s != 0*w1

        # They form a `double IPNS'
        random_pnt = up(rng.standard_normal(3))
        doubledp = (random_pnt | quadric | dualise_point(random_pnt))
        assert doubledp(0) == doubledp  # Not 0 but is a scalar
        assert (w0 | quadric | w0s) == 0 * w1  # The cone passes through the origin

        # Now let's do the construction from R(4,4) As a computational framework
        # Let's make a sphere
        sphere_quad = (w1s^w1) + (w2s^w2) + (w3s^w3) - (w0s^w0)

        # Let's try rotating, it should be invariant under rotation
        axis = np.random.randn(3)
        rotation_biv = axis[0] * (e23 - e2b3b) + axis[1] * (e1b3b - e13) + axis[2] * (e12 - e1b2b)
        Rr = np.e ** (-rotation_biv)
        np.testing.assert_allclose((Rr * sphere_quad * ~Rr).value, sphere_quad.value,
                                   rtol=1E-4, atol=1E-4)

        # Test points on the sphere surface
        for i in range(10):
            vec = np.random.randn(3)
            pnt = up(vec/np.linalg.norm(vec))
            pnts = dualise_point(pnt)
            np.testing.assert_allclose((pnt | sphere_quad | pnts).value, 0,
                                       rtol=1E-4, atol=1E-6)

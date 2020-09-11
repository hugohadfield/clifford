from functools import reduce
import operator
import textwrap

import pytest
import numpy as np
import numpy.testing
from IPython.lib import pretty

from clifford import Cl, randomMV, Frame, \
    conformalize, grade_obj, MultiVector, MVArray

import clifford


def equivalent_up_to_scale(a, b):
    return (a / b).grades() == {0}


# using fixtures here results in them only being created if needed
@pytest.fixture(scope='module')
def g2():
    return Cl(2)[0]


@pytest.fixture(scope='module')
def g3():
    return Cl(3)[0]


@pytest.fixture(scope='module')
def g4():
    return Cl(4)[0]


@pytest.fixture(scope='module')
def g5():
    return Cl(5)[0]


@pytest.fixture(scope='module')
def g4_1():
    return Cl(4, 1)[0]


@pytest.fixture(scope='module')
def g3c(g3):
    return conformalize(g3)[0]


class TestClifford:
    @pytest.fixture(params=[3, 4, 5], ids='Cl({})'.format)
    def algebra(self, request, g3, g4, g5):
        return {3: g3, 4: g4, 5: g5}[request.param]

    def test_inverse(self, algebra):
        layout, blades = algebra, algebra.blades
        a = 1. + blades['e1']
        with pytest.raises(ValueError):
            1 / a
        for i in range(10):
            a = randomMV(layout, grades=[0, 1])
            denominator = (a(1)**2)[()]-(a[()]**2)
            if abs(denominator) > 1.e-5:
                a_inv = (-a(0)/denominator) + ((1./denominator) * a(1))
                assert abs((a * a_inv)-1.) < 1.e-11
                assert abs((a_inv * a)-1.) < 1.e-11
                assert abs(a_inv - 1./a) < 1.e-11

    def test_pseudoscalar(self, algebra):
        """
        Check if the outer product of the basis elements
        is a linear multiple of the pseudoscalar
        """
        layout, blades = algebra, algebra.blades
        ps = reduce(clifford.op, layout.blades_of_grade(1))
        ps2 = layout.pseudoScalar
        assert np.linalg.matrix_rank(np.column_stack((ps.value, ps2.value))) == 1

    def test_pow_0(self, algebra):
        layout, blades = algebra, algebra.blades
        e1 = blades['e1']
        ret = e1**0
        assert type(ret) is type(e1)
        assert ret == 1
        assert ret.value.dtype == e1.value.dtype

    def test_grade_masks(self, algebra):
        layout, blades = algebra, algebra.blades
        A = layout.randomMV()
        for i in range(layout.dims + 1):
            np.testing.assert_almost_equal(A(i).value, A.value*layout.grade_mask(i))

    def test_rotor_mask(self, algebra):
        layout, blades = algebra, algebra.blades
        rotor_m = layout.rotor_mask
        rotor_m_t = np.zeros(layout.gaDims)
        for _ in range(10):
            rotor_m_t += 100*np.abs(layout.randomRotor().value)
        np.testing.assert_almost_equal(rotor_m_t > 0, rotor_m)

    def test_exp(self, g3):
        layout, blades = g3, g3.blades
        e12 = blades['e12']
        theta = np.linspace(0, 100 * np.pi, 101)
        a_list = [np.e**(t * e12) for t in theta]
        for a in a_list:
            np.testing.assert_almost_equal(abs(a), 1.0, 5)

    def test_exp_g4(self, g4):
        '''
        a numerical test for the exponential of a bivector. truth was
        generated by results of clifford v0.82
        '''
        layout, blades = g4, g4.blades

        valB = np.array([
            -0.                 ,  0.                 ,  # noqa
             0.                 , -0.                 ,  # noqa
            -0.                 , -1.9546896043012914 ,  # noqa
             0.7069828848351363 , -0.22839793693302957,  # noqa
             1.0226966962560002 ,  1.8673816483342143 ,  # noqa
            -1.7694566455296474 , -0.                 ,  # noqa
            -0.                 ,  0.                 ,  # noqa
            -0.                 , -0.                    # noqa
        ])
        valexpB = np.array([
            -0.8154675764311629  ,  0.                  ,  # noqa
             0.                  ,  0.                  ,  # noqa
             0.                  ,  0.3393508714682218  ,  # noqa
             0.22959588097548828 , -0.1331099867581965  ,  # noqa
            -0.01536404898029994 ,  0.012688721722814184,  # noqa
             0.35678394795928464 ,  0.                  ,  # noqa
             0.                  ,  0.                  ,  # noqa
             0.                  , -0.14740840378445502    # noqa
        ])

        B = layout.MultiVector(valB)
        expB = layout.MultiVector(valexpB)
        np.testing.assert_almost_equal(np.exp(B)[0].value, expB.value)

    def test_inv_g4(self, g4):
        '''
        a numerical test for the inverse of a MV. truth was
        generated by results of clifford v0.82
        '''
        layout, blades = g4, g4.blades
        valA = np.array([
            -0.3184271488037198 , -0.8751064635010213 ,  # noqa
            -1.5011710376191947 ,  1.7946332649746224 ,  # noqa
            -0.8899576254164621 , -0.3297631748225678 ,  # noqa
             0.04310366054166925,  1.3970365638677635 ,  # noqa
            -1.545423393858595  ,  1.7790215501876614 ,  # noqa
             0.4785341530609175 , -1.32279679741638   ,  # noqa
             0.5874769077573831 , -1.0227287710873676 ,  # noqa
             1.779673249468527  , -1.5415648119743852    # noqa
        ])

        valAinv = np.array([
             0.06673424072253006 , -0.005709960252678998,  # noqa
            -0.10758540037163118 ,  0.1805895938775471  ,  # noqa
             0.13919236400967427 ,  0.04123255613093294 ,  # noqa
            -0.015395162562329407, -0.1388977308136247  ,  # noqa
            -0.1462160646855434  , -0.1183453106997158  ,  # noqa
            -0.06961956152268277 ,  0.1396713851886765  ,  # noqa
            -0.02572904638749348 ,  0.02079613649197489 ,  # noqa
            -0.06933660606043765 , -0.05436077710009021    # noqa
        ])

        A = MultiVector(layout=layout, value=valA)
        Ainv = MultiVector(layout=layout, value=valAinv)

        np.testing.assert_almost_equal(A.inv().value, Ainv.value)

    def test_indexing(self, g3):
        layout, blades = g3, g3.blades
        e12 = blades['e12']
        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']
        assert e12[e12] == 1
        assert e12[e3] == 0
        with pytest.raises(ValueError):
            e12[1 + e12]
        assert e12[(2, 1)] == -1

    def test_add_float64(self, g3):
        '''
        test array_wrap method to take control addition from numpy array
        '''
        layout, blades = g3, g3.blades
        e1 = blades['e1']

        np.float64(1) + e1
        assert 1 + e1 == np.float64(1) + e1

        assert 1 + e1 == e1 + np.float64(1)

    def _random_value_array(self, layout, Nrows, Ncolumns):
        value_array = np.zeros((Nrows, Ncolumns, layout.gaDims))
        for i in range(Nrows):
            for j in range(Ncolumns):
                value_array[i, j, :] = layout.randomMV().value
        return value_array

    def test_2d_mv_array(self, g3):
        layout, blades = g3, g3.blades
        Nrows = 2
        Ncolumns = 3
        value_array_a = self._random_value_array(g3, Nrows, Ncolumns)
        value_array_b = self._random_value_array(g3, Nrows, Ncolumns)

        mv_array_a = MVArray.from_value_array(layout, value_array_a)
        assert mv_array_a.shape == (Nrows, Ncolumns)
        mv_array_b = MVArray.from_value_array(layout, value_array_b)
        assert mv_array_b.shape == (Nrows, Ncolumns)

        # check properties of the array are preserved (no need to check both a and b)
        np.testing.assert_array_equal(mv_array_a.value, value_array_a)
        assert mv_array_a.value.dtype == value_array_a.dtype
        assert type(mv_array_a.value) == type(value_array_a)

        # Check addition
        mv_array_sum = mv_array_a + mv_array_b
        array_sum = value_array_a + value_array_b
        np.testing.assert_array_equal(mv_array_sum.value, array_sum)

        # Check elementwise gp
        mv_array_gp = mv_array_a * mv_array_b
        value_array_gp = np.zeros((Nrows, Ncolumns, layout.gaDims))
        for i in range(Nrows):
            for j in range(Ncolumns):
                value_array_gp[i, j, :] = layout.gmt_func(value_array_a[i, j, :], value_array_b[i, j, :])
        np.testing.assert_array_equal(mv_array_gp.value, value_array_gp)

        # Check elementwise op
        mv_array_op = mv_array_a ^ mv_array_b
        value_array_op = np.zeros((Nrows, Ncolumns, layout.gaDims))
        for i in range(Nrows):
            for j in range(Ncolumns):
                value_array_op[i, j, :] = layout.omt_func(value_array_a[i, j, :], value_array_b[i, j, :])
        np.testing.assert_array_equal(mv_array_op.value, value_array_op)

        # Check elementwise ip
        mv_array_ip = mv_array_a | mv_array_b
        value_array_ip = np.zeros((Nrows, Ncolumns, layout.gaDims))
        for i in range(Nrows):
            for j in range(Ncolumns):
                value_array_ip[i, j, :] = layout.imt_func(value_array_a[i, j, :], value_array_b[i, j, :])
        np.testing.assert_array_equal(mv_array_ip.value, value_array_ip)

    def test_array_control(self, g3):
        '''
        test methods to take control addition from numpy arrays
        '''
        layout, blades = g3, g3.blades
        e1 = blades['e1']
        e3 = blades['e3']
        e12 = blades['e12']

        for i in range(100):

            number_array = np.random.rand(4)

            output = e12+(e1*number_array)
            output2 = MVArray([e12+(e1*n) for n in number_array])
            np.testing.assert_almost_equal(output, output2)

            output = e12 + (e1 * number_array)
            output2 = MVArray([e12 + (e1 * n) for n in number_array])
            np.testing.assert_almost_equal(output, output2)

            output = (number_array*e1) + e12
            output2 = MVArray([(n*e1) + e12 for n in number_array])
            np.testing.assert_almost_equal(output, output2)

            output = number_array / e12
            output2 = MVArray([n / e12 for n in number_array])
            np.testing.assert_almost_equal(output, output2)

            output = (e1 / number_array)
            output2 = MVArray([(e1/n) for n in number_array])
            np.testing.assert_almost_equal(output, output2)

            output = ((e1 / number_array)*e3)/e12
            output2 = MVArray([((e1 / n)*e3)/e12 for n in number_array])
            np.testing.assert_almost_equal(output, output2)

    def test_array_overload(self, algebra):
        '''
        test overload operations
        '''
        layout, blades = algebra, algebra.blades
        test_array = MVArray([layout.randomMV() for i in range(100)])

        normed_array = test_array.normal()
        other_array = np.array([t.normal().value for t in test_array])
        np.testing.assert_array_equal(normed_array.value, other_array)

        dual_array = test_array.dual()
        other_array_2 = np.array([t.dual().value for t in test_array])
        np.testing.assert_array_equal(dual_array.value, other_array_2)

    def test_comparison_operators(self, g3):
        layout, blades = g3, g3.blades
        e1 = blades['e1']
        e2 = blades['e2']

        pytest.raises(TypeError, operator.lt, e1, e2)
        pytest.raises(TypeError, operator.le, e1, e2)
        pytest.raises(TypeError, operator.gt, e1, e2)
        pytest.raises(TypeError, operator.ge, e1, e2)

        assert operator.eq(e1, e1) is True
        assert operator.eq(e1, e2) is False
        assert operator.ne(e1, e1) is False
        assert operator.ne(e1, e2) is True

        assert operator.eq(e1, None) is False
        assert operator.ne(e1, None) is True

    def test_layout_comparison_operators(self, g3, g4):
        l3a = g3
        l3b, _ = Cl(3)  # need a new copy here
        l4 = g4

        assert operator.eq(l3a, l3b) is True
        assert operator.eq(l3a, l4) is False
        assert operator.eq(l3a, None) is False

        assert operator.ne(l3a, l3b) is False
        assert operator.ne(l3a, l4) is True
        assert operator.ne(l3a, None) is True

    def test_mv_str(self, g3):
        """ Test the __str__ magic method """
        layout, blades = g3, g3.blades
        e1 = blades['e1']
        e2 = blades['e2']
        e12 = blades['e12']

        assert str(e1) == "(1^e1)"
        assert str(1 + e1) == "1 + (1^e1)"
        assert str(-e1) == "-(1^e1)"
        assert str(1 - e1) == "1 - (1^e1)"

        mv = layout.scalar * 1.0
        mv[(1,)] = float('nan')
        mv[(1, 2)] = float('nan')
        assert str(mv) == "1.0 + (nan^e1) + (nan^e12)"

    def test_nonzero(self, algebra):
        layout, blades = algebra, algebra.blades
        e1 = blades['e1']

        assert bool(e1)
        assert not bool(0*e1)

        # test nan too
        nan = float('nan')
        mv = layout.scalar * 1.0
        mv[()] = float('nan')

        # allow the nan comparison without warnings
        with np.errstate(invalid='ignore'):
            assert bool(mv) == bool(nan)  # be consistent with the builtin

    @pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
    @pytest.mark.parametrize('func', [
        operator.add,
        operator.sub,
        operator.mul,
        operator.xor,  # outer product
        operator.or_,  # inner product
    ])
    def test_binary_op_preserves_dtype(self, dtype, func, g3):
        """ test that simple binary ops on blades do not promote types """
        layout, blades = g3, g3.blades
        e1 = blades['e1'].astype(dtype)
        e2 = blades['e2'].astype(dtype)
        assert func(e1, np.int8(1)).value.dtype == dtype
        assert func(e1, e2).value.dtype == dtype

    @pytest.mark.parametrize('func', [
        operator.inv,
        operator.pos,
        operator.neg,
        MultiVector.gradeInvol,
        MultiVector.dual,
        MultiVector.right_complement,
        MultiVector.left_complement,
    ])
    def test_unary_op_preserves_dtype(self, func, g3):
        """ test that simple unary ops on blades do not promote types """
        layout, blades = g3, g3.blades
        e1 = blades['e1']
        assert func(e1).value.dtype == e1.value.dtype

    def test_indexing_blade_tuple(self, g3):
        # gh-151
        layout, blades = g3, g3.blades
        mv = layout.MultiVector(value=np.arange(2**3) + 1)

        # one swap makes the sign flip
        assert mv[1, 2] == -mv[2, 1]
        assert mv[2, 3] == -mv[3, 2]
        assert mv[3, 1] == -mv[1, 3]

        assert mv[1, 2, 3] == -mv[2, 1, 3]
        assert mv[1, 2, 3] == -mv[1, 3, 2]

        # two swaps does not
        assert mv[1, 2, 3] == mv[2, 3, 1] == mv[3, 1, 2]
        assert mv[3, 2, 1] == mv[2, 1, 3] == mv[1, 3, 2]

        # three swaps does
        assert mv[1, 2, 3] == -mv[3, 2, 1]

    def test_normalInv(self, g3):
        layout, blades = g3, g3.blades
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        assert (2*e1).normalInv() == (0.5*e1)

        with pytest.raises(ValueError):
            (0*e1).normalInv()  # divide by 0

        with pytest.raises(ValueError):
            (1 + e1 + e2).normalInv()  # mixed even and odd grades

        # produces garbage, but doesn't crash
        (1 + e1 + e2).normalInv(check=False)

        # check that not requiring normalInv works fine
        assert (1 + e1 + e2).inv() == -1 + e1 + e2

    def test_blades_list(self, g3c):
        e1 = g3c.blades['e1']
        e2 = g3c.blades['e2']
        e3 = g3c.blades['e3']
        einf = g3c.einf
        # Include a null vector just to check.
        # Note that einf is not a basis blade, so does not appear in the list.
        mv = e1 + einf + e2^einf
        assert sum(mv.blades_list) == mv

    def test_even_odd(self, g3):
        mv = g3.MultiVector(string='1 + e1 + e3 + e12 + e13 + e123')
        assert mv.even == g3.MultiVector(string='1 + e12 + e13')
        assert mv.odd == g3.MultiVector(string='e1 + e3 + e123')

    def test_commutator(self, g3):
        e1 = g3.blades['e1']
        e2 = g3.blades['e2']
        e3 = g3.blades['e3']
        e12 = g3.blades['e12']
        e13 = g3.blades['e13']
        e23 = g3.blades['e23']
        e123 = g3.blades['e123']
        assert e12.anticommutator(e23) == 0
        assert e12.commutator(e23) == e13
        assert e23.commutator(e12) == -e13

        assert e1.commutator(e123) == 0
        assert e1.anticommutator(e123) == e23
        assert e123.anticommutator(e1) == e23

    def test_basis(self, g4):
        e1 = g4.blades['e1']
        e2 = g4.blades['e2']
        e3 = g4.blades['e3']
        e4 = g4.blades['e4']
        vectors = [
            e1 + e2,
            e1 + 2*e3,
            e2 + 2*e4
        ]
        for i in range(len(vectors)):
            blade = reduce(clifford.operator.op, vectors[i:])
            basis = blade.basis()
            roundtrip = reduce(clifford.operator.op, basis)

            # Should be linear multiples of each other, seems we make
            # no guarantees about sign, magnitude, or orthogonality
            assert equivalent_up_to_scale(roundtrip, blade)

    def test_join(self, g5):
        e1 = g5.blades['e1']
        e2 = g5.blades['e2']
        e3 = g5.blades['e3']
        e4 = g5.blades['e4']
        e5 = g5.blades['e5']

        a = e5
        b = e1 + e4
        c = e1 + e3
        d = e2 + e1
        e = e4 + e2
        # chosen specifically to satisfy...
        assert len(((a^b^c)*(a^d^e)).grades()) > 2

        assert equivalent_up_to_scale(g5.scalar.join(1), 1)
        assert equivalent_up_to_scale((a).join(1), a)
        assert equivalent_up_to_scale((a).join(a), a)

        assert equivalent_up_to_scale((a).join(a), a)
        assert equivalent_up_to_scale((a).join(b), a^b)
        assert equivalent_up_to_scale((a^b).join(b^c), a^b^c)
        assert equivalent_up_to_scale((a).join(b^c), a^b^c)
        assert equivalent_up_to_scale((a^b).join(c), a^b^c)

        # should be commutative up to scale
        assert equivalent_up_to_scale((a).join(a^b), a^b)
        assert equivalent_up_to_scale((a^b).join(a), a^b)

        # need this test to avoid a fast-path
        assert equivalent_up_to_scale((a^b^c).join(a^d^e), a^b^c^d^e)

    def test_meet(self, g3):
        e1 = g3.blades['e1']
        e2 = g3.blades['e2']
        e3 = g3.blades['e3']

        assert equivalent_up_to_scale((g3.scalar).meet(g3.scalar), g3.scalar)
        assert equivalent_up_to_scale((e1^e2).meet(e2^e3), e2)

        a, b, c = (e1 + e2), (e1 + e3), (e2 + e3)
        assert equivalent_up_to_scale((a^b).meet(b^c), b)
        assert equivalent_up_to_scale((a^b^c).meet(b^c), b^c)

        assert equivalent_up_to_scale((a).meet(b^c), 1)
        assert equivalent_up_to_scale((b^c).meet(a), 1)
        assert equivalent_up_to_scale((a).meet(a), a)


class TestBasicConformal41:
    def test_metric(self, g4_1):
        layout = g4_1
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        e4 = layout.blades['e4']
        e5 = layout.blades['e5']

        assert (e1 * e1)[()] == 1
        assert (e2 * e2)[()] == 1
        assert (e3 * e3)[()] == 1
        assert (e4 * e4)[()] == 1
        assert (e5 * e5)[()] == -1

    def test_vee(self, g3c):
        layout = g3c
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        up = layout.up

        A = up(e1)
        B = up(e2)
        C = up(-e1)
        D = up(e3)

        sph = A^B^C^D
        pl = A^B^C^layout.einf

        assert sph & pl == 2*(A^B^C)
        assert pl & sph == -2*(A^B^C)

    def test_factorise(self, g3c):
        layout = g3c
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        e4 = layout.blades['e4']
        e5 = layout.blades['e5']

        up = layout.up

        blade = up(e1 + 3*e2 + 4*e3)^up(5*e1 + 3.3*e2 + 10*e3)^up(-13.1*e1)

        basis, scale = blade.factorise()
        new_blade = (reduce(lambda a, b: a^b, basis)*scale)
        print(new_blade)
        print(blade)
        np.testing.assert_almost_equal(new_blade.value, blade.value, 5)

    def test_gp_op_ip(self, g4_1):
        layout = g4_1
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        e4 = layout.blades['e4']
        e5 = layout.blades['e5']

        e123 = layout.blades['e123']
        np.testing.assert_almost_equal(e123.value, (e1 ^ e2 ^ e3).value)
        np.testing.assert_almost_equal(e123.value, (e1 * e2 * e3).value)

        e12345 = layout.blades['e12345']
        np.testing.assert_almost_equal(e12345.value, (e1 ^ e2 ^ e3 ^ e4 ^ e5).value)
        np.testing.assert_almost_equal(e12345.value, (e1 * e2 * e3 * e4 * e5).value)

        e12 = layout.blades['e12']
        np.testing.assert_almost_equal(-e12.value, (e2 ^ e1).value)

        t = np.zeros(32)
        t[0] = -1
        np.testing.assert_almost_equal(t, (e12*e12).value)

    def test_categorization(self, g3):
        layout = g3
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']

        blades = [
            layout.scalar,
            e1,
            e1 ^ e2,
            (e1 + e2) ^ e2,
        ]
        for b in blades:
            # all invertible blades are also versors
            assert b.isBlade()
            assert b.isVersor()

        versors = [
            1 + (e1^e2),
            e1 + (e1^e2^e3),
        ]
        for v in versors:
            assert not v.isBlade()
            assert v.isVersor()

        neither = [
            layout.scalar*0,
            1 + e1,
            1 + (e1^e2^e3)
        ]
        for n in neither:
            assert not n.isBlade()
            assert not n.isVersor()

    def test_blades_of_grade(self, g3):
        layout = g3
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        assert layout.blades_of_grade(0) == [layout.scalar]
        assert layout.blades_of_grade(1) == [e1, e2, e3]
        assert layout.blades_of_grade(2) == [e1^e2, e1^e3, e2^e3]
        assert layout.blades_of_grade(3) == [e1^e2^e3]


class TestBasicSpaceTime:
    def test_initialise(self):

        # Dirac Algebra  `D`
        D, D_blades = Cl(1, 3, names='d', firstIdx=0)

        # Pauli Algebra  `P`
        P, P_blades = Cl(3, names='p')

        # put elements of each in namespace
        locals().update(D_blades)
        locals().update(P_blades)


class TestBasicAlgebra:

    def test_gp_op_ip(self, g3):
        layout = g3
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']

        print('outer product')
        e123 = layout.blades['e123']
        np.testing.assert_almost_equal(e123.value, (e1 ^ e2 ^ e3).value)
        np.testing.assert_almost_equal(e123.value, (e1 * e2 * e3).value)

        print('outer product ordering')
        e12 = layout.blades['e12']
        np.testing.assert_almost_equal(-e12.value, (e2 ^ e1).value)

        print('outer product zeros')
        np.testing.assert_almost_equal(0, (e1 ^ e1).value)
        np.testing.assert_almost_equal(0, (e2 ^ e2).value)
        np.testing.assert_almost_equal(0, (e3 ^ e3).value)

        print('scalar outer product')
        np.testing.assert_almost_equal(((1 + 0 * e1) ^ (1 + 0 * e1)).value, (1 + 0 * e1).value)

        print('scalar inner product')
        np.testing.assert_almost_equal(((1 + 0 * e1) | (1 + 0 * e1)).value, 0)

    @pytest.fixture(
        params=[0, 1, 2],
        ids=['Cl(3)', 'Cl(4)', 'conformal Cl(3)']
    )
    def algebra(self, request, g3, g4, g3c):
        return [g3, g4, g3c][request.param]

    def test_grade_obj(self, algebra):
        layout = algebra
        for i in range(len(layout.sig)+1):
            mv = layout.randomMV()(i)
            assert i == grade_obj(mv)

    def test_left_multiplication_matrix(self, algebra):
        layout = algebra
        for i in range(1000):
            mv = layout.randomMV()
            mv2 = layout.randomMV()
            np.testing.assert_almost_equal(np.matmul(layout.get_left_gmt_matrix(mv), mv2.value), (mv*mv2).value)

    def test_right_multiplication_matrix(self, algebra):
        layout = algebra
        for i in range(1000):
            a = layout.randomMV()
            b = layout.randomMV()
            b_right = layout.get_right_gmt_matrix(b)
            res = a*b
            res2 = layout.MultiVector(value=b_right@a.value)
            np.testing.assert_almost_equal(res.value, res2.value)


class TestPrettyRepr:
    """ Test ipython pretty printing, with tidy line wrapping """
    def test_layout(self, g3):
        expected = textwrap.dedent("""\
        Layout([1, 1, 1],
               ids=BasisVectorIds.ordered_integers(3),
               order=BasisBladeOrder.shortlex(3),
               names=['', 'e1', 'e2', 'e3', 'e12', 'e13', 'e23', 'e123'])""")
        assert pretty.pretty(g3) == expected

    def test_multivector(self, g2):
        p = g2.MultiVector(np.arange(4, dtype=np.int32))
        assert pretty.pretty(p) == repr(p)

        expected = textwrap.dedent("""\
        MultiVector(Layout([1, 1],
                           ids=BasisVectorIds.ordered_integers(2),
                           order=BasisBladeOrder.shortlex(2),
                           names=['', 'e1', 'e2', 'e12']),
                    [0, 1, 2, 3],
                    dtype=int32)""")

        # ipython printing only kicks in in ugly mode
        try:
            clifford.ugly()
            assert pretty.pretty(p) == expected
        finally:
            clifford.pretty()

    def test_multivector_predefined(self):
        """ test the short printing of predefined layouts """
        from clifford.g2 import layout as g2
        p_i = g2.MultiVector(np.arange(4, dtype=np.int32))
        p_f = g2.MultiVector(np.arange(4, dtype=np.float64))
        assert pretty.pretty(p_i) == repr(p_i)
        assert pretty.pretty(p_f) == repr(p_f)

        # float is implied
        expected_i = "clifford.g2.layout.MultiVector([0, 1, 2, 3], dtype=int32)"
        expected_f = "clifford.g2.layout.MultiVector([0.0, 1.0, 2.0, 3.0])"

        # ipython printing only kicks in in ugly mode
        try:
            clifford.ugly()
            assert pretty.pretty(p_i) == expected_i
            assert pretty.pretty(p_f) == expected_f
        finally:
            clifford.pretty()


class TestFrame:

    def check_inv(self, A):
        Ainv = None
        for k in range(3):
            try:
                Ainv = A.inv
            except ValueError:
                pass
        if Ainv is None:
            return
        for m, a in enumerate(A):
            for n, b in enumerate(A.inv):
                if m == n:
                    assert(a | b == 1)
                else:
                    assert(a | b == 0)

    @pytest.mark.parametrize(('p', 'q'), [
        (2, 0), (3, 0), (4, 0)
    ])
    def test_frame_inv(self, p, q):
        layout, blades = Cl(p, q)
        A = Frame(layout.randomV(p + q))
        self.check_inv(A)

    @pytest.mark.parametrize(('p', 'q'), [
        (2, 0), (3, 0), (4, 0)
    ])
    def test_innermorphic(self, p, q):
        layout, blades = Cl(p, q)

        A = Frame(layout.randomV(p+q))
        R = layout.randomRotor()
        B = Frame([R*a*~R for a in A])
        assert A.is_innermorphic_to(B)

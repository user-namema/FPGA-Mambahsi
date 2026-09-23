import unittest
from dataclasses import replace
import torch
from fixed_gpu_backend import update_wide_gpu, certify_wide
from ssm_error_ablation import SSMNumericConfig, integer_update


class WideIntegerTests(unittest.TestCase):
    def test_wide_matches_arbitrary_precision(self):
        # Includes products wider than INT64, mixed signs and state saturation.
        torch.manual_seed(14)
        h=torch.randint(-(1<<31),(1<<31)-1,(500,),dtype=torch.int64)
        a=torch.randint(0,1<<24,(500,),dtype=torch.int64)
        kbu=torch.randint(-(1<<40),1<<40,(500,),dtype=torch.int64)
        for rounding in ('single','separate'):
            c=replace(SSMNumericConfig(),rounding=rounding)
            for fraction in (20,24,30):
                actual=update_wide_gpu(h,a,kbu,fraction,c)
                expected,_=integer_update(h,a,kbu,fraction,c,native_safe=False)
                self.assertTrue(torch.equal(actual,expected),(rounding,fraction))

    def test_half_ties_and_cancellation(self):
        for rounding in ('single','separate'):
            c=replace(SSMNumericConfig(),a_fraction_bits=4,state_fraction_bits=4,rounding=rounding)
            h=torch.tensor([-24,-8,8,24,31,-31,17,-17],dtype=torch.int64)
            a=torch.ones_like(h)
            k=torch.tensor([0,0,0,0,-15,15,-8,8],dtype=torch.int64)
            actual=update_wide_gpu(h,a,k,7,c)
            expected,_=integer_update(h,a,k,7,c,native_safe=False)
            self.assertTrue(torch.equal(actual,expected))

    def test_reject_unsafe_container(self):
        with self.assertRaises(OverflowError):
            certify_wide(dict(a=torch.tensor([1<<24]),k=torch.tensor([1<<40]),k_fraction_bits=0),SSMNumericConfig())

if __name__=='__main__':unittest.main()

import unittest
from run_max_init_qat_fpga import simulator_device


class DeviceTests(unittest.TestCase):
    def test_indexed_cuda_is_mapped_for_simulator(self):
        self.assertEqual(simulator_device('cuda:0',{}),('cuda',{'CUDA_VISIBLE_DEVICES':'0'}))
        self.assertEqual(simulator_device('cuda:1',{'CUDA_VISIBLE_DEVICES':'3,7'})[1]['CUDA_VISIBLE_DEVICES'],'7')
        self.assertEqual(simulator_device('cuda:0',{'CUDA_VISIBLE_DEVICES':'GPU-abc'})[1]['CUDA_VISIBLE_DEVICES'],'GPU-abc')

    def test_cpu_plain_cuda_and_invalid_index(self):
        for device in ('cpu','cuda'):
            self.assertEqual(simulator_device(device,{})[0],device)
        with self.assertRaises(ValueError):simulator_device('cuda:2',{'CUDA_VISIBLE_DEVICES':'0'})
        with self.assertRaises(ValueError):simulator_device('cuda:-1',{})


if __name__=='__main__':unittest.main()

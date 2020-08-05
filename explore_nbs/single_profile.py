import fastai_xla_extensions.core
from fastai2.vision.all import *
from my_timesaver_utils.profiling import *
path = untar_data(URLs.PETS)/'images'
Path.BASE_PATH = path; path.ls()
print(f'running on default_device() & cuda is {torch.cuda.is_available()}')

img = PILImage.create(path/'Abyssinian_1.jpg')
resize = Resize(size=200)
img2 = resize(img,split_idx=0)




timg2 = TensorImage(array(img2)).permute(2,0,1).float()/255.

def batch_ex(bs, device): return TensorImage(timg2[None].to(device).expand(bs, *timg2.shape))


b768_img = batch_ex(768, default_device()); (b768_img.shape, b768_img.device)


flip_tfm = Flip(p=1.0)
# run without profile
run_with_profile = True
F.grid_sample = profile_call(F.grid_sample) if run_with_profile else F.grid_sample

@profile_call
def mtest(b_img):
    #set_trace()
    new_b_img = flip_tfm(b_img)
    return new_b_img
    
clear_prof_data()
print("--- 10 image tensor loops:")
for i in range(10):
    print("--- ---------------------------------")
    new_b768_img = mtest(b768_img)
print("--- ")
print_prof_data()
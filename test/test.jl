using Test

h5file = "/home/nwknoblauch/Dropbox/Repos/ptb_workflowr/data/ngwas_df.h5"
file = h5open(h5file,"r")
zvec = read(file,"gwas/z");
regid = read(file,"gwas/region_id");
close(file);

h5file = "/home/nwknoblauch/Dropbox/Repos/ptb_workflowr/data/annotations.h5"
file = h5open(h5file,"r")
rowvec = read(file,"anno/SNP");
colvec = read(file,"anno/feature");
close(file);

# max_reg = maximum(regid)

# h5file = "/home/nwknoblauch/Dropbox/Repos/ptb_workflowr/data/split_d.h5"
# file = h5open(h5file,"r")
# sub_d = [read(file,string(i)) for i in 1:max_reg]
# close(file);


z_grp = split_array(zvec,regid);



za = [-0.596142502520553, 0.226281864987637, -0.000726228309287968];
prior = [1e-3,1e-3,1e-3];
pip = [0.0,0.0,0.0];
ba = compute_log10_BF(za);
res = EM_update!(pip,ba,prior)
tret = -0.00075799
pip_r=np.array([0.000451804,0.000405431,0.000398141])



zscores = [0.992034011766786, -0.259309620006531, 0.98160170428947]
true_bf = [-0.246520744871551, -0.389501683458334, -0.249769243036791]
@test(isapprox(compute_log10_BF(zscores),true_bf))

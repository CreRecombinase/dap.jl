module utils
export split_array,compute_log10_BF,calc_log10_lik,EM_update!

using HDF5
using StatsBase
using LinearAlgebra



function split_array(array,split_idx)
    im = indexmap(split_idx);
    max_k = maximum(keys(im))
    im[max_k+1]=length(array)+1
    return [view(array,im[i]:(im[i+1]-1)) for i in 1:max_k]
end




function compute_log10_BF(zscores)
    kv = [1,4,16,25]
    kva = reshape([0.5*log(1/(1+x)) for x in kv],(1,4))
    kvb = reshape([0.5*(x/(1+x)) for x in kv],(1,4))
    kvbz = broadcast(*,(zscores.*zscores),kvb);
    kvab = broadcast(+,kva,kvbz);
    kvr = kvab/log(10);
    kvrmax = maximum(kvr,dims=2);
    kvrmm=@. 0.25*(10^(kvr-kvrmax));
    return (kvrmax+log10.(sum(kvrmm,dims=2)))[:,1];
end


function calc_log10_lik(bfv,prior_a,p0=0)
    max_el = max(maximum(bfv),0)
    return log10(dot(prior_a,(10 .^ (bfv .- max_el) )) + p0 * 10 ^ (-max_el) ) + max_el
end



function EM_update!(pip::AbstractArray{Float64,1},bf::AbstractArray{Float64,1},prior::AbstractArray{Float64,1})
    locus_pi0 = prod(1 .- prior)
    pip = prior ./ (1 .- prior)
    if locus_pi0 < 1e-100
        locus_pi0 = 1e-100
    end
    pip *= locus_pi0
    loglik = calc_log10_lik(bf,pip,locus_pi0)
    pip = @. 10 ^ ( log10(pip) + bf -loglik)
    return loglik
end

end

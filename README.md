# DiliLazyKV



## 1. Install Packages
## 2 Two-stage method
### 2.1 Diligent Head and Lazy Head Identification

We employ a Needle-in-a-Haystack test to identify heads crucial for different capabilities. The Diligent
head’s DiliScore and the Lazy head’s LazyScore jointly constitute the Inference Score (InfScore).

```python
python structure_head_InfScore.py 
```

### 2.2 Collaborative Layer-Head KV Cache Budget Allocation

During the prefill stage, we allocate the KV cache budget for
each head based on the inter-layer aggregation and intra-layer distribution of InfScore variance across heads.

```
max_capacity_prompts=128

devices=(0 1) 
head_choices=('reason') #  copy, reason
betas=(1.351) 
counter=0
for((i=0;i<${#head_choices[@]};i++));do 
    for((j=0;j<${#betas[@]};j++));do # This loop will also run once as betas has 1 element
        device=${devices[counter]}
        head_choice=${head_choices[i]}
        beta=${betas[j]}
        temp=1
        nohup bash head_base.sh \
            $device \
            ReasonKV \
            ${max_capacity_prompts} \
            flash_attention_2 \
            meta-llama/Meta-Llama-3-8B-Instruct \
            $head_choice \
            $beta \
            $temp > ./longbench_logs/llama3_ReasoniKV_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
        ((counter+=1))
    done
done
```


## Acknowledgement
Thanks [HeadKV-R2](https://github.com/FYYFU/HeadKV) for providing open-source code and data.
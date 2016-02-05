
-- misc utilities

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function table_size(t)
    local size = 0
    for _ in pairs(t) do size = size + 1 end
    return size
end

function check_vocab_compatible(old_vocab, new_vocab)
    -- old_vocab \subseteq new_vocab
    for c,i in pairs(old_vocab) do 
        if not new_vocab[c] == i then 
            return false, -1
        end
    end
    local num_new = 0
    for w in pairs(new_vocab) do 
        if old_vocab[w] == nil then
            num_new = num_new + 1
        end
    end
    return true, num_new
end

function load_model(path)
    local checkpoint = torch.load(path)

    if checkpoint.vocab_mapping then
        local vocab_compatible, num_new = check_vocab_compatible(checkpoint.vocab_mapping, qb.vocab_mapping)
        assert(vocab_compatible, 'error, the vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
        print('%d new words', num_new)
    end


    if checkpoint.ans_mapping then
        local ans_compatible, num_new = check_vocab_compatible(checkpoint.ans_mapping, qb.ans_mapping)
        assert(ans_compatible, 'error, the answer mapping for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
        assert(num_new == 0, 'error, the answer mapping for this dataset has ' .. num_new .. ' new answers/classes. This is trouble.')
    end

    return checkpoint
end


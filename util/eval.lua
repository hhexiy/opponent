local eval = {}

-- compute accuracy
function eval.accuracy(pred, gold, mask)
    -- isSameSizeAs require same type too!
    assert(pred:isSameSizeAs(gold) and gold:isSize(mask:size()), 'dimension mismatch')
    local correct = pred:maskedSelect(mask):eq(gold:maskedSelect(mask))
    return correct:sum() / correct:size(1) 
end

function eval.seq_accuracy(pred, gold, mask)
    local correct = 0
    local total = 0
    for i=1,pred:size(1) do 
        local len = mask[i]:sum()
        if len > 0 then
            total = total + 1
            if pred[i][len] == gold[i][len] then
                correct = correct + 1
            end
        end
    end
    return correct / total
end

function eval.max_seq_accuracy(pred, gold, mask)
    local correct = 0
    local total = 0 
    for i=1,pred:size(1) do 
        if mask[i][1] > 0 then
            total = total + 1
            if pred[i]:maskedSelect(mask[i]):eq(gold[i][1]):sum() > 0 then
                correct = correct + 1
            end
        end
    end
    return correct / total
end

-- predict the majority class
function eval.constant_baseline(gold, mask, const_pred)
    local correct  = gold:maskedSelect(mask):eq(const_pred)
    return correct:sum() / correct:size(1) 
end

-- predict the average buzz position
function eval.average_baseline(gold, mask)
    average_buzz_pos = math.ceil((gold:eq(1):sum() / gold:size(1)) + 1)
    --print('average buzz pos:' .. average_buzz_pos)
    pred = torch.Tensor(gold:size()):fill(2)
    pred:narrow(2, 1, average_buzz_pos):fill(1)
    return eval.accuracy(pred, gold, mask)
end

function eval.get_payoff(my_buzz_pos, my_correct, human_buzzes)
    local my_payoff = 0
    local total = 0
    for j=1,#human_buzzes do
        local human_buzz_pos = human_buzzes[j][2]
        local human_correct = human_buzzes[j][3]
        if human_buzz_pos < my_buzz_pos then
            if human_correct == 1 then
                my_payoff = my_payoff - 10
            else
                my_payoff = my_payoff + (my_correct == 1 and 15 or 5)
            end
        elseif human_buzz_pos > my_buzz_pos then
            if my_correct == 1 then
                my_payoff = my_payoff + 10
            else
                my_payoff = my_payoff - (human_correct == 1 and 15 or 5)
            end
        else
            if my_correct == 1 then
                my_payoff = my_payoff + (human_correct == 1 and 0 or 15)
            else
                my_payoff = my_payoff - (human_correct == 0 and 0 or 15) 
            end
        end  
        total = total + 1
    end
    return my_payoff, total
end

function eval.predicted_buzz(ans_preds, buzz_preds, gold, mask, qids, buzzes)
    local my_payoff = 0
    local total = 0
    local buzz_pos_sum = 0
    for i=1,ans_preds:size(1) do
        if mask[i][1] > 0 then 
            local my_buzz_pos = mask[i]:sum()
            for j=1,my_buzz_pos-1 do
                if buzz_preds[i][j] == qb.BUZZ then
                    my_buzz_pos = j
                    break
                end
            end
            local my_correct = ans_preds[i][my_buzz_pos] == gold[i][1] and 1 or 0
            local buzz = buzzes[qids[i]]
            if buzz ~= nil then
                local payoff, tot = eval.get_payoff(my_buzz_pos, my_correct, buzz)
                my_payoff = my_payoff + payoff
                total = total + tot
                buzz_pos_sum = buzz_pos_sum + my_buzz_pos * tot
            end
        end
    end
    return my_payoff / total, buzz_pos_sum / total
end

-- buzz at the most confident position
function eval.max_margin_buzz(logprobs, gold, mask, qids, buzzes)
    -- logprobs: size * seq_length * output_size
    --local probs = logprobs:exp()
    local sorted_probs, sorted_ans = torch.sort(logprobs, 3, true)
    -- margin: size * seq_length
    local margin = (sorted_probs:narrow(3, 1, 1) - sorted_probs:narrow(3, 2, 1)):squeeze(3)
    assert(margin:lt(0):sum() == 0)
    local _, buzz_pos = margin:max(2)

    local my_payoff = 0
    local total = 0
    local buzz_pos_sum = 0
    for i=1,logprobs:size(1) do
        if mask[i][1] > 0 then 
            local my_buzz_pos = buzz_pos[i][1]
            local my_correct = sorted_ans[i][my_buzz_pos][1] == gold[i][1] and 1 or 0
            local buzz = buzzes[qids[i]]
            if buzz ~= nil then
                local payoff, tot = eval.get_payoff(my_buzz_pos, my_correct, buzz)
                my_payoff = my_payoff + payoff
                total = total + tot
                buzz_pos_sum = buzz_pos_sum + my_buzz_pos * tot
            end
        end
    end
    return my_payoff / total, buzz_pos_sum / total
end

return eval

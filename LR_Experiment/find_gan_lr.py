def log_keeping(avg_loss,beta,loss,best_loss, batch_num,losses, log_lrs ):
    #Compute the smoothed loss
    avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
    smoothed_loss = avg_loss / (1 - beta**batch_num)
    #Stop if the loss is exploding
    if batch_num > 1 and smoothed_loss > 4 * best_loss:
        return log_lrs, losses
    #Record the best loss
    if smoothed_loss < best_loss or batch_num==1:
        best_loss = smoothed_loss
    #Store the values
    losses.append(smoothed_loss)
    log_lrs.append(math.log10(lr))
    

def find_gan_lr( ds_man, init_value = 1e-8, final_value=10., beta = 0.98):
    #  cost_operation, optimize_operation


    num = len(ds_man.num_examples)-1
    mult = (final_value / init_value) ** (1/num)
    lr_disc = init_value
    lr_gen = init_value

    batch_num = 0

    gen_avg_loss = 0.
    gen_best_loss = 0.
    gen_losses = []
    gen_log_lrs = []

    disc_avg_loss = 0.
    disc_best_loss = 0.
    disc_losses = []
    disc_log_lrs = []
        
    with tf.Session() as sess:

        for data in range(len(ds_man.num_examples)):
            batch_num += 1
            #As before, get the loss for this mini-batch of inputs/outputs
            

            gen_loss = sess.run(gen_cost)

            batch_data = ds_man.next_batch(256)
            disc_loss = sess.run(disc_cost, feed_dict={real_data: batch_data} )
                    
            # DISC
            log_keeping(disc_avg_loss, beta, disc_loss, disc_best_loss, batch_num, disc_losses, disc_log_lrs )
            # GEN 

            log_keeping(gen_avg_loss,beta,loss,gen_best_loss, batch_num, gen_losses, gen_log_lrs )
            #Do one optimization step

            # train critic
            for i_ in range(5):
                batch_data = ds_man.next_batch(256)
                disc_cost_ = sess.run( disc_train_op, feed_dict={real_data: batch_data,disc_lr: lr} )

            sess.run(gen_train_op,  feed_dict={gen_lr: lr} )
        
            #Update the lr for the next step
            lr_disc *= mult
            lr_gen *= mult
            
        return gen_log_lrs, gen_losses, disc_log_lrs, disc_losses
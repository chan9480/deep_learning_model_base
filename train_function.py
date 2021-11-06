# 모델 체크포인트 저장
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

@tf.function
def train_step(images):
  '''
  배치 한번당 일어나는 학습
  '''
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #모델 사용
    generated_images = generator(noise, training=True)
    #loss계산에 필요한 값
    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)
    #loss 계산
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
  # gradients 계산
  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  #오치 역전파, 가중치 업데이트
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def train(dataset, epochs):
  '''
  input :
  dataset은 tf의 데이터셋 객체여야하며, epochs는 전체데이터 학습횟수
  
  train_step이라는 함수로 1배치당 학습을 따로 정의하여 
  이 함수에서 실행시킨다.
  '''
  for epoch in range(epochs):
    start = time.time() #start는 1 epochs마다 걸린시간(맨처음부터 누적된 시간)을 담는다.

    for batch in dataset:
      train_step(batch)

    # GIF를 위한 이미지를 바로 생성합니다.
    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed)

    # 15 에포크가 지날 때마다 모델을 저장합니다.
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # 마지막 에포크가 끝난 후 생성합니다.
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)


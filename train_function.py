import tensorflow as tf
import os

# 모델 정의 (함수형이든 sequential 이든 맘대루)
''' 함수형'''
input_ = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(input_)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_ = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(input_, output_)
'''sequential 1'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2, activation="relu"))
model.add(tf.keras.layers.Dense(3, activation="relu"))
model.add(tf.keras.layers.Dense(4))

'''sequential 2'''
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(2, activation="relu"),
        tf.keras.layers.Dense(3, activation="relu"),
        tf.keras.layers.Dense(4),
    ]
)

#loss function 정의 (예시 sparse categorical crossentropy)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

# optimizer 정의 (예시 Adam)
optimizer = tf.keras.optimizers.Adam()

# metric 정의
train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

val_loss = tf.keras.metrics.Mean()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

# train_step 정의
@tf.function
def train_step(x, y):
    '''
    배치 한번에 대한 학습과정을 커스터마이징
    '''
    with tf.GradientTape() as tape:
        # 1. 모델사용 예측 (prediction)
        predictions = model(x)
        # 2. Loss 계산
        loss = loss_function(y, predictions)
    
    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # loss와 accuracy를 업데이트 합니다.
    train_loss(loss)
    train_acc(y, predictions)

#validation_step 정의
@tf.function
def val_step(x, y):
    '''
    1 epoch가 끝나면 test step으로 loss를 계산한다.
    '''
    # 1. 예측 (prediction)
    predictions = model(x)
    # 2. Loss 계산
    loss = loss_function(y, predictions)
    
    # Val셋에 대해서는 gradient를 계산 및 backpropagation 하지 않습니다.
    
    # loss와 accuracy를 업데이트 합니다.
    val_loss(loss)
    val_acc(y, predictions)


# 모델 체크포인트 저장
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                model=model)
'''============================================================================================'''
# train 정의
def train(dataset, epochs, batch_size=64, val = 0.2, patience = 5, best_loss=1e10, n=1):
  '''
  input :
    dataset은 tf의 tf.data 객체여야하며,
    epochs는 전체데이터 학습횟수
    val은 검증데이터 비율
    patience 는 early stopping을 판단할때, 연속적으로 성능향상이 일어나지 않은 횟수가 patience값이 되면 break.
    best_loss 값보다 val_loss가 작으면 계속해서 val_loss로 업데이트
    n 번마다 학습결과를 print함
    batch_size 는 batch size
  
  설명 :
  이 함수에서 train_step, test_step 등 실행시킨다.
  또한 특정 epoch마다 수행할 일을 정의할 수 있다.
  early stopping을 구현해놓음.
  '''
  len_dataset = int(dataset.__len__())    # batch로 묶기전 길이 shuffle buffer에 쓸거임.
  for epoch in range(epochs):
    #start는 1 epochs마다 걸린시간(맨처음부터 누적된 시간)을 담는다.
    start = time.time() 

    # cross validation
    dataset = dataset.shuffle(buffer_size=len_dataset,
                               reshuffle_each_iteration=False).batch(batch_size) #shuffle, batch를 적용해줌. buffer는 shuffle로 뽑을 애들의 후보의 크기랄까 
    len_iter = int(dataset.__len__()) #batch로 묶인상태의 길이
    len_val_iter = int(len_iter*val)      #val 비율을 곱함
    validation_dataset = dataset.take(len_val_iter)
    train_dataset = dataset.skip(len_val_iter)

    # 1batch씩 train함.
    for train_batch_x, train_batch_y in train_dataset: 
      train_step(train_batch_x, train_batch_y)
    
    # test셋에 대해서 test함
    for val_batch_x, val_batch_y in validation_dataset:
      val_step(val_batch_x, val_batch_y)
    
    # n에포크 마다 결과들을 print한다.
    if (epoch + 1) % n == 0:
      template = '에포크: {}, 손실: {:.5f}, 정확도: {:.2f}%, 검증 손실: {:.5f}, 검증 정확도: {:.2f}%'
      print (template.format(epoch+1,
                            train_loss.result(),
                            train_acc.result()*100,
                            val_loss.result(),
                            val_acc.result()*100))
      print ('소요시간 {} sec'.format(time.time()-start))
    # 15 에포크가 지날 때마다 모델을 저장합니다.
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    # early stopping 구현. (예시: epoch연속으로 patience 수만큼 val_loss감소가 없다면 STOP!!)
    wait += 1
    if val_loss < best:
      best = val_loss
      wait = 0
    if wait >= patience:
      print ('요청epochs {} 번 중 {} 번 완료후 early stopping.'.format(epochs, epoch+1))
      break
  # 마지막 에포크가 끝난 후 할일 이 아래에 작성
  print ('요청epochs {} 번을 모두 완료후 종료됨.'.format(epochs))

'''-------------------------------------------------------------------------------------------'''
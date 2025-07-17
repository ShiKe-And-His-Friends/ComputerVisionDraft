#include <linux/delay.h>
#include <linux/gpio.h>
#include <linux/interrupt.h>
#include <linux/module.h>

#include <linux/module.h>	//所有模块都需要的头文件
#include <linux/kernel.h>
#include <linux/fs.h>	//文件系统有关的，结构体file_operations也在fs头文件定义
#include <linux/init.h>	//init和exit相关宏
#include <linux/delay.h>
#include <linux/irq.h>
#include <asm/uaccess.h>	//linux中的用户态内存交互函数，copy_from_user(),copy_to_user()等
#include <asm/irq.h>	//linux中断定义
#include <asm/io.h>
#include <linux/sched.h>	//声明printk()这个内核态的函数
#include <linux/interrupt.h>	//包含与中断相关的大部分宏及结构体的定义，request_irq()等
#include <linux/device.h>	//
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/gfp.h>
#include <linux/jiffies.h>  // 用于时间戳处理

// 定义消抖时间间隔（毫秒）
#define MS_TO_JIFFIES(ms) ((ms) * HZ / 1000)

// 定义GPIO配置结构体
struct gpio_desc {
    unsigned int gpio_num;       // GPIO编号
    unsigned int irq_type;       // 中断类型
    unsigned int debounce_time;  // 消抖时间(ms)
    unsigned long last_time;     // 上次触发时间
    unsigned char base_key_val;  
    unsigned char key_val;       // 键值
    spinlock_t lock;             // 每个GPIO独立的锁
};

// 定义多个GPIO描述符
static struct gpio_desc gpio_descs[] = {
    // 原有GPIO2_6 (1200ms)
    { .gpio_num = 2*8 + 6, .irq_type = IRQF_TRIGGER_RISING, .debounce_time = 1200,.base_key_val=0x00 ,.key_val = 0x00 },
    
    // 新增GPIO3_5 (上升沿+下降沿, 20ms)
    { .gpio_num = 3*8 + 5, .irq_type = IRQF_TRIGGER_RISING | IRQF_TRIGGER_FALLING, .debounce_time = 1,.base_key_val=0x04 ,.key_val = 0x00 },
    
    // 新增GPIO4_7 (上升沿+下降沿, 30ms)
    { .gpio_num = 4*8 + 7, .irq_type = IRQF_TRIGGER_RISING | IRQF_TRIGGER_FALLING, .debounce_time = 1,.base_key_val=0x08 , .key_val = 0x00 },

    // 原有GPIO0_0 (1200ms)
    //{ .gpio_num = 0*8 + 0, .irq_type = IRQF_TRIGGER_RISING, .debounce_time = 1200,.base_key_val=0x0B ,.key_val = 0x00 },

    // 原有GPIO5_2 (1200ms)
    { .gpio_num = 5*8 + 2, .irq_type = IRQF_TRIGGER_RISING, .debounce_time = 1200,.base_key_val=0x0E ,.key_val = 0x00 },
    
};

#define GPIO_DESC_CNT (sizeof(gpio_descs) / sizeof(gpio_descs[0]))

// 中断事件标志, 中断服务程序将它置1，btn_drv_read将它清0 
static volatile int ev_press = 0;
static struct fasync_struct *button_async;  // 定义一个结构
static DECLARE_WAIT_QUEUE_HEAD(button_waitq);
static struct class *btndrv_class;
static struct device *btndrv_dev;
int major;

static irqreturn_t gpio_dev_test_isr(int irq, void *dev_id)
{
    struct gpio_desc *desc = (struct gpio_desc *)dev_id;
    unsigned long flags;
    unsigned long current_time;
    irqreturn_t ret = IRQ_NONE;
    int gpio_val;

    current_time = jiffies;
    
    // 关中断并加锁（保护共享变量）
    spin_lock_irqsave(&desc->lock, flags);

    // 检查消抖时间
    if (time_after(current_time, desc->last_time + MS_TO_JIFFIES(desc->debounce_time))) {
        gpio_val = gpio_get_value(desc->gpio_num);
        printk(KERN_DEBUG "[%s %d] Valid interrupt on GPIO%d_%d (value=%d)\n",
                __func__, __LINE__, desc->gpio_num/8, desc->gpio_num%8, gpio_val);
        
        // 更新键值（根据边沿类型设置不同值）
        if (desc->irq_type & IRQF_TRIGGER_RISING && gpio_val){
            desc->key_val = 0x02;  // 上升沿
        }
        else if (desc->irq_type & IRQF_TRIGGER_FALLING && !gpio_val)
        {
            desc->key_val = 0x01;  // 下降沿
        }
            
        ev_press = 1;
        desc->last_time = current_time;
        
        // 解锁后再唤醒队列，减少锁持有时间
        spin_unlock_irqrestore(&desc->lock, flags);
        
        wake_up_interruptible(&button_waitq);
        kill_fasync(&button_async, SIGIO, POLL_IN);
        ret = IRQ_HANDLED;
    } else {
        // 解锁
        spin_unlock_irqrestore(&desc->lock, flags);
        printk(KERN_DEBUG "[%s %d] Debounce: Ignored interrupt on GPIO%d_%d\n",
                __func__, __LINE__, desc->gpio_num/8, desc->gpio_num%8);
    }

    return ret;
}

/*********************************************************************************************************
*功能：模块打开函数
*参数：
*返回值：无
**********************************************************************************************************/
static int btn_drv_open(struct inode *inode, struct file *file)
{
    printk(KERN_DEBUG "driver: btn_drv open\n");  
    return 0;
}

/*********************************************************************************************************
*功能：模块读取函数
*参数：
*返回值：无
**********************************************************************************************************/
ssize_t btn_drv_read(struct file *file, char __user *buf, size_t size, loff_t *ppos)
{
    struct gpio_desc *desc = NULL;
    int i;
    unsigned char key_info = 0;  // [GPIO编号, 键值]
    int found = 0;
    unsigned long flags;

    if (size != 1)
        return -EINVAL;

    if(file->f_flags & O_NONBLOCK){       /* 非 阻塞操作 */  
        // 非阻塞模式：快速检查所有GPIO
        for (i = 0; i < GPIO_DESC_CNT; i++) {
            spin_lock_irqsave(&gpio_descs[i].lock, flags);
            if (gpio_descs[i].key_val) {
                // 复制数据并清除键值
                key_info = gpio_descs[i].base_key_val + gpio_descs[i].key_val;
                gpio_descs[i].key_val = 0;
                found = 1;
            }
            spin_unlock_irqrestore(&gpio_descs[i].lock, flags);
            
            if (found)
                break;
        }
        
        if (!found) {
            return -EAGAIN;  // 没有按键事件
        }
    
        printk(KERN_DEBUG "driver: no-block button %02x \n", key_info); 

        if (copy_to_user(buf, &key_info, 1))
            return -EFAULT;
    } else {
        // 如果没有按键动作, 休眠
        wait_event_interruptible(button_waitq, ev_press);
        
        // 原子操作：检查并清除所有按键状态
        spin_lock_irq(&button_waitq.lock);
        if (ev_press) {
            // 查找触发的GPIO并复制数据
            for (i = 0; i < GPIO_DESC_CNT; i++) {
                if (gpio_descs[i].key_val) {
                    key_info = gpio_descs[i].base_key_val + gpio_descs[i].key_val;
                    gpio_descs[i].key_val = 0;
                    found = 1;
                    break;
                }
            }
            
            // 只有所有按键都处理完才清除全局标志
            if (found) {
                for (i = 0; i < GPIO_DESC_CNT; i++) {
                    if (gpio_descs[i].key_val) {
                        found = 2;  // 表示还有其他按键未处理
                        break;
                    }
                }
                
                if (found == 1)  // 所有按键都已处理
                    ev_press = 0;
            }
        }
        spin_unlock_irq(&button_waitq.lock);
        
        if (!found) {
            return -EIO;  // 理论上不会发生
        }
    
        printk(KERN_DEBUG "driver: block button %02x \n", key_info); 

        if (copy_to_user(buf, &key_info, 1))
            return -EFAULT;
    }
  
    return 1;
}

/*********************************************************************************************************
*功能：模块关闭函数
*参数：
*返回值：无
**********************************************************************************************************/
int btn_drv_close(struct inode *inode, struct file *file)
{
    printk(KERN_DEBUG "driver: btn_drv close\n");  
    return 0;
}

/*********************************************************************************************************
*功能：模块软件轮循函数
*参数：
*返回值：无
**********************************************************************************************************/
static unsigned btn_drv_poll(struct file *file, poll_table *wait)
{
    unsigned int mask = 0;
    int i;

    poll_wait(file, &button_waitq, wait);

    // 检查所有GPIO的key_val
    for (i = 0; i < GPIO_DESC_CNT; i++) {
        spin_lock(&gpio_descs[i].lock);
        if (gpio_descs[i].key_val) {
            mask |= POLLIN | POLLRDNORM;
            spin_unlock(&gpio_descs[i].lock);
            break;
        }
        spin_unlock(&gpio_descs[i].lock);
    }

    return mask;
}

/*********************************************************************************************************
*功能：模块异步通知函数
*参数：
*返回值：无
**********************************************************************************************************/
static int btn_drv_fasync(int fd, struct file *filp, int on)   
{
    printk(KERN_DEBUG "driver: btn_drv_fasync\n");
    return fasync_helper(fd, filp, on, &button_async);
}

static struct file_operations g_key_event_fops = {
    .owner   =  THIS_MODULE,
    .open    =  btn_drv_open,     
    .read    =  btn_drv_read,     
    .release =  btn_drv_close,
    .poll    =  btn_drv_poll,
    .fasync  =  btn_drv_fasync,
};

static int __init gpio_dev_test_init(void)
{
    int i, ret;

    printk(KERN_DEBUG "[%s %d]\n", __func__, __LINE__);
    
    // 初始化等待队列和异步通知
    init_waitqueue_head(&button_waitq);
    button_async = NULL;
    
    // 初始化每个GPIO描述符和锁
    for (i = 0; i < GPIO_DESC_CNT; i++) {
        spin_lock_init(&gpio_descs[i].lock);  // 为每个GPIO初始化独立的锁
        gpio_descs[i].last_time = 0;
        
        // 请求GPIO
        if (gpio_request(gpio_descs[i].gpio_num, NULL)) {
            printk(KERN_DEBUG "[%s %d]gpio_request fail! gpio_num=%d\n", 
                   __func__, __LINE__, gpio_descs[i].gpio_num);
            ret = -EIO;
            goto err_free_gpios;
        }
        
        // 设置为输入
        if (gpio_direction_input(gpio_descs[i].gpio_num)) {
            printk(KERN_DEBUG "[%s %d]gpio_direction_input fail! gpio_num=%d\n", 
                   __func__, __LINE__, gpio_descs[i].gpio_num);
            ret = -EIO;
            goto err_free_gpios;
        }
        
        // 注册中断
        ret = request_irq(gpio_to_irq(gpio_descs[i].gpio_num), 
                          gpio_dev_test_isr, 
                          gpio_descs[i].irq_type | IRQF_SHARED,
                          "gpio_dev_test", &gpio_descs[i]);
        if (ret) {
            printk(KERN_DEBUG "[%s %d]request_irq fail! gpio_num=%d\n", 
                   __func__, __LINE__, gpio_descs[i].gpio_num);
            goto err_free_gpios;
        }
    }

    // 注册字符设备和创建设备节点
    major = register_chrdev(0, "btn_drv", &g_key_event_fops);
    if (major < 0) {
        printk(KERN_DEBUG "[%s %d]register_chrdev fail!\n", __func__, __LINE__);
        ret = major;
        goto err_free_irqs;
    }

    btndrv_class = class_create(THIS_MODULE, "btn_drv");
    if (IS_ERR(btndrv_class)) {
        printk(KERN_DEBUG "[%s %d]class_create fail!\n", __func__, __LINE__);
        ret = PTR_ERR(btndrv_class);
        goto err_unregister_chrdev;
    }

    btndrv_dev = device_create(btndrv_class, NULL, MKDEV(major, 0), NULL, "buttons");
    if (IS_ERR(btndrv_dev)) {
        printk(KERN_DEBUG "[%s %d]device_create fail!\n", __func__, __LINE__);
        ret = PTR_ERR(btndrv_dev);
        goto err_destroy_class;
    }

    return 0;

err_destroy_class:
    class_destroy(btndrv_class);
err_unregister_chrdev:
    unregister_chrdev(major, "btn_drv");
err_free_irqs:
    for (i = 0; i < GPIO_DESC_CNT; i++) {
        if (gpio_is_valid(gpio_descs[i].gpio_num))
            free_irq(gpio_to_irq(gpio_descs[i].gpio_num), &gpio_descs[i]);
    }
err_free_gpios:
    for (i = 0; i < GPIO_DESC_CNT; i++) {
        if (gpio_is_valid(gpio_descs[i].gpio_num))
            gpio_free(gpio_descs[i].gpio_num);
    }
    return ret;
}

static void __exit gpio_dev_test_exit(void)
{
    int i;
    
    printk(KERN_DEBUG "[%s %d]\n", __func__, __LINE__);

    // 释放设备和类
    device_unregister(btndrv_dev);
    class_destroy(btndrv_class);
    unregister_chrdev(major, "btn_drv");

    // 释放中断和GPIO
    for (i = 0; i < GPIO_DESC_CNT; i++) {
        free_irq(gpio_to_irq(gpio_descs[i].gpio_num), &gpio_descs[i]);
        gpio_free(gpio_descs[i].gpio_num);
    }
}

module_init(gpio_dev_test_init);
module_exit(gpio_dev_test_exit);
MODULE_DESCRIPTION("Multi-GPIO Interrupt Driver with Debounce");
MODULE_LICENSE("GPL");
from tkinter import *

from PIL import Image, ImageTk

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)

        # main window/frame
        self.master = master

        self.init_window()

    def init_window(self):
        
        self.master.title("GUI")

        self.pack(fill=BOTH, expand=1)

        #quit_button = Button(
        #    self,
        #    text="Quit",
        #    command=self.client_exit
        #   )
        #quit_button.place(x=0, y=0)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label='Save')
        file.add_command(label='Exit', command=self.client_exit)

        menu.add_cascade(label='File', menu=file)

        edit = Menu(menu)
        edit.add_command(label='Show Image', command=self.show_image)
        edit.add_command(label='Show Text', command=self.show_text)
        edit.add_command(label='Undo')
        menu.add_cascade(label='Edit', menu=edit)
        
        self.ment = StringVar()
        entry = Entry(self.master, textvariable=self.ment)
        entry.place(x = 200, y = 100)
        entry.pack()

    def show_text(self):
        text = Label(self, text=self.ment.get())
        text.pack()
    
    def show_image(self):
        load = Image.open("image/pic.png")
        render = ImageTk.PhotoImage(load)

        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)
    
    def client_exit(self):
        exit()
    
root = Tk()
root.geometry("400x300")

app = Window(root)

root.mainloop()

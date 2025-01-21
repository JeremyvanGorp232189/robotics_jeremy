{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import random\n",
    "\n",
    "# Add the path to sim_class.py to the Python path\n",
    "sys.path.append(r'datalab_tasks\\task9\\Y2B-2023-OT2_Twin')\n",
    "\n",
    "from sim_class import Simulation\n",
    "\n",
    "# Initialize the simulation with one robot\n",
    "sim = Simulation(num_agents=1)\n",
    "\n",
    "# Main simulation loop for 1000 steps\n",
    "for step in range(1000):\n",
    "    # Generate random actions for the robot\n",
    "    velocity_x = random.uniform(-0.5, 0.5)\n",
    "    velocity_y = random.uniform(-0.5, 0.5)\n",
    "    velocity_z = random.uniform(-0.5, 0.5)\n",
    "    drop_command = random.randint(0, 1)\n",
    "\n",
    "    # Actions: [velocity_x, velocity_y, velocity_z, drop_command]\n",
    "    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]\n",
    "\n",
    "    # Run one step in the simulation and capture the state\n",
    "    state = sim.run(actions)\n",
    "    print(f\"Step {step}: Robot State: {state}\")\n",
    "\n",
    "# Reset the simulation after completion\n",
    "sim.reset(num_agents=1)\n",
    "print(\"Simulation reset completed.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
